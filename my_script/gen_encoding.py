# import torch
# import psycopg2
# import json
# import pandas as pd
# import os
# from model.database_util import Encoding, formatJoin, formatFilter

# # 数据库连接参数
# DB_CONFIG = {
#     "database": "genome",
#     "user": "xuyining",
#     "host": "localhost",
#     "port": "5433"
# }

# def generate_encoding_clean(plan_csv_path, save_path):
#     conn = psycopg2.connect(**DB_CONFIG)
#     cur = conn.cursor()

#     if not os.path.exists(plan_csv_path):
#         print(f"Error: {plan_csv_path} not found.")
#         return

#     df = pd.read_csv(plan_csv_path)
    
#     # 集合用于去重存储
#     used_columns = set()   # 存储格式 "alias.column"
#     alias_to_table = {}    # 存储格式 "alias": "table_name"
#     node_types = set()
#     join_strs = set()

#     print("Step 1: Scanning plans to extract unique Schema information...")
#     for plan_json in df['json']:
#         plan_dict = json.loads(plan_json)
        
#         def traverse(node):
#             # 1. 记录算子类型
#             node_types.add(node['Node Type'])
            
#             # 2. 建立 别名 -> 物理表 的映射（为了去数据库查统计信息）
#             if 'Relation Name' in node and 'Alias' in node:
#                 alias_to_table[node['Alias']] = node['Relation Name']
            
#             # 3. 提取连接条件（调用项目自带的 formatJoin）
#             js = formatJoin(node)
#             if js:
#                 join_strs.add(js)
            
#             # 4. 提取谓词列（完全模拟 Encoding.encode_filters 的解析逻辑）
#             filters, alias = formatFilter(node)
#             if filters and alias:
#                 for f in filters:
#                     # 剥离括号，按 AND 分割
#                     f_clean = ''.join(c for c in f if c not in '()')
#                     for part in f_clean.split(' AND '):
#                         # 拿到列名，组合成 alias.column
#                         col_name = part.split(' ')[0]
#                         used_columns.add(f"{alias}.{col_name}")
            
#             if 'Plans' in node:
#                 for sub in node['Plans']:
#                     traverse(sub)
        
#         traverse(plan_dict['Plan'])

#     print(f"Found {len(used_columns)} columns, {len(node_types)} operators, and {len(alias_to_table)} tables.")

#     # 准备 Encoding 构造参数
#     column_min_max_vals = {}
#     col2idx = {"NA": 0}
    
#     print("Step 2: Fetching MIN/MAX values from Database...")
#     # 排序以保证生成的 ID 具有确定性
#     sorted_cols = sorted(list(used_columns))
#     for idx, col_key in enumerate(sorted_cols):
#         col2idx[col_key] = idx + 1 # ID 从 1 开始
        
#         alias, col = col_key.split('.')
#         # 找到该别名对应的真实物理表名
#         table = alias_to_table.get(alias, alias)
        
#         try:
#             cur.execute(f"SELECT MIN({col}), MAX({col}) FROM {table}")
#             res = cur.fetchone()
#             if res and res[0] is not None:
#                 column_min_max_vals[col_key] = [float(res[0]), float(res[1])]
#             else:
#                 column_min_max_vals[col_key] = [0.0, 1.0]
#                 print(f"  Warning: No data for {col_key}, using default [0, 1]")
#         except Exception as e:
#             print(f"  Error fetching {col_key}: {e}")
#             column_min_max_vals[col_key] = [0.0, 1.0]
#             conn.rollback()

#     # 5. 初始化 Encoding 对象
#     # 注意：Encoding 类内部会自动处理 idx2col, idx2op 等反向映射
#     encoding = Encoding(column_min_max_vals, col2idx)
    
#     print("Step 3: Populating ID vocabularies for Tables, Joins, and Types...")
#     # 填充算子 ID
#     for nt in node_types:
#         encoding.encode_type(nt)
#     # 填充连接 ID
#     for js in join_strs:
#         encoding.encode_join(js)
#     # 填充物理表 ID
#     for table_name in alias_to_table.values():
#         encoding.encode_table(table_name)

#     # 6. 保存为 .pt 文件
#     torch.save({'encoding': encoding}, save_path)
#     print(f"Successfully saved encoding to {save_path}")
    
#     cur.close()
#     conn.close()

# if __name__ == "__main__":
#     # 配置你的路径
#     PLAN_PATH = '/data2/xuyining/QueryFormer/data/genome/train_plan.csv'
#     SAVE_PATH = '/data2/xuyining/QueryFormer/data/genome/encoding.pt'
#     generate_encoding_clean(PLAN_PATH, SAVE_PATH)


import torch
import psycopg2
import json
import pandas as pd
import os
from model.database_util import Encoding, formatJoin, formatFilter

# 数据库连接参数
DB_CONFIG = {
    "database": "ergastf1",
    "user": "vipuser",
    "host": "localhost",
    "port": "5432"
}

def is_number(s):
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False

def generate_encoding_automatic_v2(plan_csv_path, save_path):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    df = pd.read_csv(plan_csv_path)
    
    used_columns = set()
    alias_to_table = {}
    node_types = set()
    join_strs = set()
    ops_found = set(['NA']) # 默认放入 NA

    print("Step 1: Scanning plans for Schema AND Operators...")
    for plan_json in df['json']:
        plan_dict = json.loads(plan_json)
        
        def traverse(node):
            node_types.add(node['Node Type'])
            if 'Relation Name' in node and 'Alias' in node:
                alias_to_table[node['Alias']] = node['Relation Name']
            
            try:
                js = formatJoin(node)
                if js: join_strs.add(js)
            except KeyError:
                # 如果节点缺少 Alias 等键导致 KeyError，直接跳过
                pass

            try: 
                filters, alias = formatFilter(node)
                if filters and alias:
                    for f in filters:
                        f_clean = ''.join(c for c in f if c not in '()')
                        for part in f_clean.split(' AND '):
                            # 分割列名、操作符、数值
                            # 比如: "kind_id >= 7" -> ["kind_id", ">=", "7"]
                            parts = part.strip().split(' ')
                            if len(parts) >= 2:
                                col_name = parts[0]
                                op = parts[1]
                                used_columns.add(f"{alias}.{col_name}")
                                ops_found.add(op) # 动态记录操作符
            except (KeyError, TypeError, ValueError):
                # 同样的，对于不支持该节点的 filter 提取逻辑也跳过
                pass
            
            if 'Plans' in node:
                for sub in node['Plans']: traverse(sub)
        
        traverse(plan_dict['Plan'])

    print(f"Found operators: {ops_found}")

    # 构建 op2idx 字典
    op2idx = {op: i for i, op in enumerate(sorted(list(ops_found)))}

    # 准备列 ID
    column_min_max_vals = {}
    col2idx = {"NA": 0}
    sorted_cols = sorted(list(used_columns))
    for idx, col_key in enumerate(sorted_cols):
        col2idx[col_key] = idx + 1
        alias, col = col_key.split('.')
        table = alias_to_table.get(alias, alias)
        pure_col = col.replace('::text', '')
        try:
            cur.execute(f"SELECT MIN({pure_col}), MAX({pure_col}) FROM {table}")
            res = cur.fetchone()
            if res and res[0] is not None:
                if is_number(res[0]) and is_number(res[1]):
                    # 如果是数字，存正常范围
                    column_min_max_vals[col_key] = [float(res[0]), float(res[1])]
                else:
                    # 如果是字符串或时间戳，打上特殊的 "CAT" 分类标记
                    print(f"  Info: Column {col_key} is categorical/string.")
                    column_min_max_vals[col_key] = ["CAT", "CAT"]
            else:
                column_min_max_vals[col_key] = ["CAT", "CAT"]
        except:
            print(f"  Warning: Error fetching {col_key}: {e}")
            column_min_max_vals[col_key] = ["CAT", "CAT"]
            conn.rollback()

    # 初始化 Encoding，传入动态生成的 op2idx
    encoding = Encoding(column_min_max_vals, col2idx, op2idx=op2idx)
    
    for nt in node_types: encoding.encode_type(nt)
    for js in join_strs: encoding.encode_join(js)
    for table in alias_to_table.values(): encoding.encode_table(table)

    torch.save({'encoding': encoding}, save_path)
    print(f"Successfully saved encoding with custom operators to {save_path}")
    cur.close()
    conn.close()

# if __name__ == "__main__":
#     PLAN_PATH = '/home/vipuser/QueryFormer/data/genome/query/train_plan.csv'
#     SAVE_PATH = '/home/vipuser/QueryFormer/data/genome/encoding.pt'
#     generate_encoding_automatic_v2(PLAN_PATH, SAVE_PATH)


if __name__ == "__main__":
    # 修改这里，扫描所有可能用到的计划文件
    PLAN_FILES = [
        '/home/vipuser/QueryFormer/data/ergastf1/query/train_plan.csv',
        '/home/vipuser/QueryFormer/data/ergastf1/query/test_plan.csv'
    ]
    SAVE_PATH = '/home/vipuser/QueryFormer/data/ergastf1/encoding.pt'
    
    # 你可以修改 generate_encoding_automatic_v2 函数使其接受一个列表
    # 或者简单粗暴地运行三次，但要把 encoding 对象在循环外初始化
    # 最稳妥的方法：合并三个 CSV 传给原来的函数
    import pandas as pd
    combined_df = pd.concat([pd.read_csv(f) for f in PLAN_FILES])
    combined_csv_path = '/tmp/combined_plans.csv'
    combined_df.to_csv(combined_csv_path, index=False)
    
    generate_encoding_automatic_v2(combined_csv_path, SAVE_PATH)