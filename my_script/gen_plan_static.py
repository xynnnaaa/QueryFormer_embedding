import psycopg2
import json
import csv

# 数据库连接配置
DB_CONFIG = {
    "host": "localhost",
    "database": "stats",
    "user": "vipuser",
    "port": 5432
}


import re
import datetime

def timestamp_to_string(timestamp_str):
    # 将时间戳字符串转换为整数
    unix_timestamp = int(timestamp_str)
    
    # 将 Unix 时间戳转换为 datetime 对象
    dt = datetime.datetime.fromtimestamp(unix_timestamp)
    
    # 将 datetime 对象格式化为时间字符串
    timestamp_str = dt.strftime('%Y-%m-%d %H:%M:%S')
    
    return timestamp_str

def convert_timestamps_in_sql(sql):
    # 使用正则表达式匹配形如 alias.column op value 的谓词，其中 column 以 date 或 Date 结尾，value 是数字
    pattern = r'(\w+\.(\w+))\s*([<>=!]+)\s*(\d+)'
    def replace_match(match):
        full_col = match.group(1)
        col_name = match.group(2)
        op = match.group(3)
        value = match.group(4)
        if col_name.endswith('date') or col_name.endswith('Date'):
            converted_value = timestamp_to_string(value)
            return f"{full_col} {op} '{converted_value}'::timestamp"
        else:
            return match.group(0)
    return re.sub(pattern, replace_match, sql)

def parse_to_sql(line):
    """
    解析原始格式: tables#joins#filters#card
    注意：这里假设最后一位是真实基数标签
    """
    parts = line.strip().split('#')
    if len(parts) < 4:
        return None, None
    
    tables = parts[0]
    joins = parts[1]
    filters_raw = parts[2].split(',')
    actual_card = float(parts[3]) # 提取最后的真实基数
    
    filter_list = []
    for i in range(0, len(filters_raw), 3):
        if i+2 >= len(filters_raw): break
        col, op, val = filters_raw[i], filters_raw[i+1], filters_raw[i+2]
        filter_list.append(f"{col} {op} {val}")
    
    where_clause = []
    if joins:
        where_clause.append(joins.replace(',', ' AND '))
    if filter_list:
        where_clause.append(' AND '.join(filter_list))
    
    sql = f"SELECT * FROM {tables}"
    if where_clause:
        sql += " WHERE " + " AND ".join(where_clause)
    
    return sql, actual_card

def generate_static_plans(input_csv, output_csv):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    with open(input_csv, 'r') as fin, open(output_csv, 'w', newline='') as fout:
        writer = csv.writer(fout)
        writer.writerow(['id', 'json'])
        
        for index, line in enumerate(fin):
            sql, actual_card = parse_to_sql(line)
            if not sql: continue

            if DB_CONFIG.get('database') == 'stats':
                sql = convert_timestamps_in_sql(sql)
            
            try:
                # 关键：不带 ANALYZE，只获取计划结构
                cur.execute(f"EXPLAIN (FORMAT JSON) {sql}")
                plan_data_list = cur.fetchone()[0] # pg返回的是一个列表
                plan_dict = plan_data_list[0] # 取出字典内容
                
                # 手动注入真实基数到 Plan 的根节点
                # QueryFormer 加载器主要看这两个字段
                plan_dict["Plan"]["Actual Rows"] = actual_card
                
                # 因为没有真正执行，Execution Time 通常不存在，手动填充为 0 或 1
                # 这样可以防止 trainer.py 计算 Cost 时出现空值
                plan_dict["Execution Time"] = 0.0 
                plan_dict["Planning Time"] = 0.0
                
                # 转换为 JSON 字符串存入 CSV
                writer.writerow([index, json.dumps(plan_dict)])
                
                if index % 100 == 0:
                    print(f"Processed {index} queries...")
                    
            except Exception as e:
                print(f"Error at ID {index}: {e}")
                conn.rollback()
                continue

    print("Static plan generation completed.")
    
    cur.close()
    conn.close()

if __name__ == "__main__":
    generate_static_plans('/home/vipuser/QueryFormer/data/stats/query/train.csv', '/home/vipuser/QueryFormer/data/stats/query/train_plan.csv')

    generate_static_plans('/home/vipuser/QueryFormer/data/stats/query/test.csv', '/home/vipuser/QueryFormer/data/stats/query/test_plan.csv')