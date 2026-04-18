import pandas as pd
import json
import numpy as np
import psycopg2
import os
import csv
from model.database_util import formatFilter # 复用项目中的解析逻辑

# 数据库连接参数 - 请根据你的环境修改
DB_CONFIG = {
    "database": "ergastf1",
    "user": "vipuser",
    "host": "localhost",
    "port": 5432
}

def to_vals(data_list):
    for dat in data_list:
        val = dat[0]
        if val is not None: break
    try:
        float(val)
        return np.array(data_list, dtype=float).squeeze()
    except:
#         print(val)
        res = []
        for dat in data_list:
            try:
                mi = dat[0].timestamp()
            except:
                mi = 0
            res.append(mi)
        return np.array(res)

def generate_hist_final(plan_csv_path, output_path):
    # 1. 连接数据库
    conn = psycopg2.connect(**DB_CONFIG)
    conn.set_session(autocommit=True)
    cur = conn.cursor()

    # 2. 从物理计划中提取 (表名, 别名, 列名)
    if not os.path.exists(plan_csv_path):
        print(f"Error: {plan_csv_path} not found.")
        return

    df_plans = pd.read_csv(plan_csv_path)
    # 集合存储 (table, alias, col_name)
    table_alias_cols = set()

    print("Scanning plans to extract table, alias, and columns...")
    
    for plan_json in df_plans['json']:
        plan = json.loads(plan_json)['Plan']
        def scan(node):
            if 'Relation Name' in node and 'Alias' in node:
                table = node['Relation Name']
                alias = node['Alias']
                filters, _ = formatFilter(node)
                if filters:
                    for f in filters:
                        f_clean = ''.join(c for c in f if c not in '()')
                        for part in f_clean.split(' AND '):
                            # 提取列名 (处理 t.id 这种带别名的和不带别名的情况)
                            raw_col = part.split(' ')[0]
                            col_name = raw_col.split('.')[-1] if '.' in raw_col else raw_col
                            table_alias_cols.add((table, alias, col_name))
            if 'Plans' in node:
                for sub in node['Plans']: scan(sub)
        scan(plan)

    # 3. 开始计算直方图
    hist_records = []
    total = len(table_alias_cols)
    print(f"Found {total} unique combinations. Starting data scan...")

    # 预先生成频率 (freq) 的 Hex 编码
    # 50个桶，每个桶频率 0.02
    num_bins = 50
    freq_array = np.full(num_bins, 1.0 / num_bins, dtype=np.float64)
    freq_hex = freq_array.tobytes().hex()

    for i, (table, alias, col) in enumerate(table_alias_cols):
        print(f"[{i+1}/{total}] Processing {table} ({alias}).{col} ...")
        
        try:
            # 这里的查询逻辑完全遵循你提供的示例
            cmd = f'SELECT {col} FROM {table} AS {alias}'
            cur.execute(cmd)
            data = cur.fetchall()
            col_array = to_vals(data)
            
            if col_array.size == 0:
                print(f"  Warning: No data for {table}.{col}, skipping.")
                continue

            # 使用百分位数计算等高桶边界 (0, 2, 4 ... 100)
            # 这会生成 51 个边界值，形成 50 个桶
            hists = np.nanpercentile(col_array, range(0, 101, 2), axis=0)
            
            # 格式化 bins 字符串：要求 " val1 val2 ... "
            bins_str = " " + " ".join([str(int(b)) for b in hists]) + " "
            
            hist_records.append({
                'table': table,
                'column': col,
                'table_alias': alias,  # 存储别名
                'table_column': f"{alias}.{col}",
                'bins': bins_str,
                'freq': freq_hex
            })
        except Exception as e:
            print(f"  Error processing {table}.{col}: {e}")
            conn.rollback()

    # 4. 保存为 CSV
    hist_df = pd.DataFrame(hist_records)
    hist_df.to_csv(output_path, index=False)
    print(f"\nSuccessfully saved final histogram file to: {output_path}")
    
    cur.close()
    conn.close()

if __name__ == "__main__":
    generate_hist_final('/home/vipuser/QueryFormer/data/ergastf1/query/train_plan.csv', '/home/vipuser/QueryFormer/data/ergastf1/histogram_string.csv')
