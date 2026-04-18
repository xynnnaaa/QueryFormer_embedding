import numpy as np
import pandas as pd
import csv
import torch
import os
import hashlib


def get_join_embedding(embedding_file, use_join_embedding=0, expected_dim=768):
    """
    加载查询级别的 Join Embedding
    返回格式: { query_id: tensor }
    """
    if use_join_embedding == 0:
        return None
        
    if embedding_file and os.path.exists(embedding_file):
        raw_data = torch.load(embedding_file)
        
        # 假设 raw_data 是一个列表、字典或 Tensor，将其转换为安全的字典格式
        join_embeddings = {}
        if isinstance(raw_data, dict):
            iterator = raw_data.items()
        else:
            iterator = enumerate(raw_data)
            
        for q_idx, vec in iterator:
            vec_np = vec.numpy() if torch.is_tensor(vec) else np.array(vec)
            join_embeddings[q_idx] = vec_np
            
        print(f"Loaded {len(join_embeddings)} Join Embeddings from {embedding_file}")
        return join_embeddings
    else:
        print(f"Error: Join Embedding file {embedding_file} not found.")
        exit(1)

def hash_to_float(val_str):
    """
    将任何字符串哈希到 [0.0, 1.0] 之间。
    这保证了同一个字符串（如 'f'）总是得到相同的特征值，
    而不同的字符串（如 'm'）得到不同的特征值。
    """
    # 取 MD5 的前 8 位 (十六进制)，转为整数后除以最大值
    hash_int = int(hashlib.md5(str(val_str).encode('utf-8')).hexdigest()[:8], 16)
    return hash_int / 0xFFFFFFFF


## bfs shld be enough
def floyd_warshall_rewrite(adjacency_matrix):
    (nrows, ncols) = adjacency_matrix.shape
    assert nrows == ncols
    M = adjacency_matrix.copy().astype('long')
    for i in range(nrows):
        for j in range(ncols):
            if i == j: 
                M[i][j] = 0
            elif M[i][j] == 0: 
                M[i][j] = 60
    
    for k in range(nrows):
        for i in range(nrows):
            for j in range(nrows):
                M[i][j] = min(M[i][j], M[i][k]+M[k][j])
    return M

def get_job_table_sample(workload_file_name, num_materialized_samples = 1000, use_single_embedding=0, embedding_file=None):

    tables = []
    samples = []

    tables_info = [] # 存储每条查询的表信息 [(table_name, alias), ...]

    # Load queries
    with open(workload_file_name + ".csv", 'r') as f:
        data_raw = list(list(rec) for rec in csv.reader(f, delimiter='#'))
        for row in data_raw:
            tables.append(row[0].split(','))

            # Store table information for each query
            table_info = []
            for table in row[0].split(','):
                table_name = table.split(' ')[0]
                alias = table.split(' ')[-1]
                table_info.append((table_name, alias))
            tables_info.append(table_info)


            if int(row[3]) < 1:
                print("Queries must have non-zero cardinalities")
                exit(1)

    print("Loaded queries with len ", len(tables))

    if use_single_embedding == 1:
        if embedding_file and os.path.exists(embedding_file):
            raw_data = torch.load(embedding_file) # { seq_id: {alias: tensor} }
            table_sample = []
            for query_id, table_info in enumerate(tables_info):
                sample_dict = {}
                for table_name, alias in table_info:
                    if alias in raw_data[query_id]:
                        sample_dict[table_name] = raw_data[query_id][alias].numpy() if torch.is_tensor(raw_data[query_id][alias]) else raw_data[query_id][alias]
                    else:
                        print(f"Alias {alias} not found in embedding file for query {query_id}")
                        sample_dict[table_name] = torch.zeros(768)  # Assuming embedding size is 768
                table_sample.append(sample_dict)

            print(f"Loaded single embedding samples from {embedding_file}")
        else:
            print(f"Embedding file {embedding_file} not found. Please provide a valid embedding file.")
            exit(1)
    else:
        # Load bitmaps
        num_bytes_per_bitmap = int((num_materialized_samples + 7) >> 3)
        with open(workload_file_name + ".bitmaps", 'rb') as f:
            for i in range(len(tables)):
                four_bytes = f.read(4)
                if not four_bytes:
                    print("Error while reading 'four_bytes'")
                    exit(1)
                num_bitmaps_curr_query = int.from_bytes(four_bytes, byteorder='little')
                bitmaps = np.empty((num_bitmaps_curr_query, num_bytes_per_bitmap * 8), dtype=np.uint8)
                for j in range(num_bitmaps_curr_query):
                    # Read bitmap
                    bitmap_bytes = f.read(num_bytes_per_bitmap)
                    if not bitmap_bytes:
                        print("Error while reading 'bitmap_bytes'")
                        exit(1)
                    bitmaps[j] = np.unpackbits(np.frombuffer(bitmap_bytes, dtype=np.uint8))
                samples.append(bitmaps)
        print("Loaded bitmaps")
        table_sample = []
        for ts, ss in zip(tables,samples):
            d = {}
            for t, s in zip(ts,ss):
                tf = t.split(' ')[0] # remove alias
                d[tf] = s
            table_sample.append(d)
            
    return table_sample


def get_hist_file(hist_path, bin_number = 50):
    hist_file = pd.read_csv(hist_path)
    for i in range(len(hist_file)):
        freq = hist_file['freq'][i]
        freq_np = np.frombuffer(bytes.fromhex(freq), dtype=float)
        hist_file['freq'][i] = freq_np

    table_column = []
    for i in range(len(hist_file)):
        alias = hist_file['table_alias'][i]
        col = hist_file['column'][i]
        combine = '.'.join([alias,col])
        table_column.append(combine)
    hist_file['table_column'] = table_column


    for rid in range(len(hist_file)):
        hist_file['bins'][rid] = \
            [int(i) for i in hist_file['bins'][rid][1:-1].split(' ') if len(i)>0]

    if bin_number != 50:
        hist_file = re_bin(hist_file, bin_number)

    return hist_file

def re_bin(hist_file, target_number):
    for i in range(len(hist_file)):
        freq = hist_file['freq'][i]
        bins = freq2bin(freq,target_number)
        hist_file['bins'][i] = bins
    return hist_file

def freq2bin(freqs, target_number):
    freq = freqs.copy()
    maxi = len(freq)-1
    
    step = 1. / target_number
    mini = 0
    while freq[mini+1]==0:
        mini+=1
    pointer = mini+1
    cur_sum = 0
    res_pos = [mini]
    residue = 0
    while pointer < maxi+1:
        cur_sum += freq[pointer]
        freq[pointer] = 0
        if cur_sum >= step:
            cur_sum -= step
            res_pos.append(pointer)
        else:
            pointer += 1
    
    if len(res_pos)==target_number: res_pos.append(maxi)
    
    return res_pos



class Batch():
    def __init__(self, attn_bias, rel_pos, heights, x, join_emb=None, y=None):
        super(Batch, self).__init__()

        self.heights = heights
        self.x, self.y = x, y
        self.attn_bias = attn_bias
        self.rel_pos = rel_pos
        self.join_emb = join_emb

    def to(self, device):

        self.heights = self.heights.to(device)
        self.x = self.x.to(device)

        self.attn_bias, self.rel_pos = self.attn_bias.to(device), self.rel_pos.to(device)

        if self.join_emb is not None:
            self.join_emb = self.join_emb.to(device)

        return self

    def __len__(self):
        return self.in_degree.size(0)


def pad_1d_unsqueeze(x, padlen):
    x = x + 1 # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_2d_unsqueeze(x, padlen):
    # dont know why add 1, comment out first
#    x = x + 1 # pad id = 0
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype) + 1
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)

def pad_rel_pos_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)

def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(float('-inf'))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)


def collator(small_set):
    y = small_set[1]
    xs = [s['x'] for s in small_set[0]]
    
    num_graph = len(y)
    x = torch.cat(xs)
    attn_bias = torch.cat([s['attn_bias'] for s in small_set[0]])
    rel_pos = torch.cat([s['rel_pos'] for s in small_set[0]])
    heights = torch.cat([s['heights'] for s in small_set[0]])

    join_embs = torch.stack([s['join_embedding'] for s in small_set[0]])
    
    return Batch(attn_bias, rel_pos, heights, x, join_emb=join_embs), y

def filterDict2Hist(hist_file, filterDict, encoding, max_filters=3):
    buckets = len(hist_file['bins'][0]) 
    empty = np.zeros(buckets - 1)
    ress = np.zeros((max_filters, buckets-1))

    num_filter = min(len(filterDict['colId']), max_filters)
    for i in range(num_filter):
        colId = filterDict['colId'][i]
        col = encoding.idx2col[colId]

        # 遇到 NA 或者该列是字符类型，直接返回 empty（全0向量
        if col == 'NA' or encoding.is_categorical_col.get(col, False):
            ress[i] = empty
            continue
        
        # 尝试去直方图文件里找，如果没找到，也安全退回 empty
        hist_row = hist_file.loc[hist_file['table_column'] == col, 'bins']
        if len(hist_row) == 0:
            ress[i] = empty
            continue

        bins = hist_row.item()

        opId = filterDict['opId'][i]
        op = encoding.idx2op[opId]
        
        val = filterDict['val'][i]
        mini, maxi = encoding.column_min_max_vals[col]
        val_unnorm = val * (maxi-mini) + mini
        
        left = 0
        right = len(bins)-1
        for j in range(len(bins)):
            if bins[j]<val_unnorm:
                left = j
            if bins[j]>val_unnorm:
                right = j
                break

        res = np.zeros(len(bins)-1)

        if op == '=':
            res[left:right] = 1
        elif op in ['!=', '<>']:
            res[:] = 1
            res[left:right] = 0
        elif op in ['<', '<=']:
            res[:left] = 1
        elif op in ['>', '>=']:
            res[right:] = 1
        ress[i] = res
    
    ress = ress.flatten()
    return ress     
        



def formatJoin(json_node):
   
    join = None
    if 'Hash Cond' in json_node:
        join = json_node['Hash Cond']
    elif 'Join Filter' in json_node:
        join = json_node['Join Filter']
    ## TODO: index cond
    elif 'Index Cond' in json_node and not json_node['Index Cond'][-2].isnumeric() and len(json_node['Index Cond']) > 2:
        join = json_node['Index Cond']
    
    ## sometimes no alias, say t.id 
    ## remove repeat (both way are the same)
    if join is not None:

        clean_join = join.replace('(', '').replace(')', '')

        twoCol = clean_join.split(' = ')

        alias = json_node.get('Alias')
        processed_cols = []
        for col in twoCol:
            col = col.strip()
            # 如果列名已经带了点号（比如 st_u.id），或者节点根本没有别名，就保持原样
            if '.' in col or alias is None:
                processed_cols.append(col)
            else:
                # 只有不带点号且存在别名时，才拼接别名
                processed_cols.append(alias + '.' + col)
        join = ' = '.join(sorted(processed_cols))

        # twoCol = [json_node['Alias'] + '.' + col 
        #           if len(col.split('.')) == 1 else col for col in twoCol ] 
        # join = ' = '.join(sorted(twoCol))
    
    return join
    
def formatFilter(plan):
    alias = None
    if 'Alias' in plan:
        alias = plan['Alias']
    else:
        pl = plan
        while 'parent' in pl:
            pl = pl['parent']
            if 'Alias' in pl:
                alias = pl['Alias']
                break
    
    filters = []
    if 'Filter' in plan:
        filters.append(plan['Filter'])
    if 'Index Cond' in plan and plan['Index Cond'][-2].isnumeric():
        filters.append(plan['Index Cond'])
    if 'Recheck Cond' in plan:
        filters.append(plan['Recheck Cond'])
        
    
    
    return filters, alias

class Encoding:
    def __init__(self, column_min_max_vals, 
                 col2idx, op2idx={'>':0, '=':1, '<':2, 'NA':3}):
        self.column_min_max_vals = column_min_max_vals
        self.col2idx = col2idx
        self.op2idx = op2idx

        self.is_categorical_col = {}
        for col, min_max in column_min_max_vals.items():
            if min_max is None or not isinstance(min_max[0], (int, float)) or min_max[0] == "CAT":
                self.is_categorical_col[col] = True
            else:
                self.is_categorical_col[col] = False
        
        idx2col = {}
        for k,v in col2idx.items():
            idx2col[v] = k
        self.idx2col = idx2col
        
        # self.idx2op = {0:'>', 1:'=', 2:'<', 3:'NA'}
        self.idx2op = {}
        for k, v in self.op2idx.items():
            self.idx2op[v] = k
        
        self.type2idx = {}
        self.idx2type = {}
        self.join2idx = {}
        self.idx2join = {}
        
        self.table2idx = {'NA':0}
        self.idx2table = {0:'NA'}
    
    def normalize_val(self, column, val, log=False):
        if self.is_categorical_col.get(column, False):
            return hash_to_float(val)

        if isinstance(val, str):
            # 1. 处理类型转换符：去掉 '::' 及其后面的所有内容
            # 示例: "157.64'::double precision" -> "157.64'"
            val = val.split('::')[0]
            
            # 2. 去掉单引号和双引号
            # 示例: "157.64'" -> "157.64"
            val = val.replace("'", "").replace('"', "")
            
            # 3. 去掉可能存在的空格
            val = val.strip()
        
        try:
            val = float(val)
            if column in self.column_min_max_vals:
                mini, maxi = self.column_min_max_vals[column]
                val_norm = 0.0
                if maxi > mini:
                    val_norm = (val-mini) / (maxi-mini)
                return val_norm
            else:
                return 0.0
        except Exception as e:
            return hash_to_float(val)
    
    def encode_filters(self, filters=[], alias=None): 
        ## filters: list of dict 

#        print(filt, alias)
        if len(filters) == 0:
            return {'colId':[self.col2idx['NA']],
                   'opId': [self.op2idx['NA']],
                   'val': [0.0]} 
        res = {'colId':[],'opId': [],'val': []}
        for filt in filters:
            filt = ''.join(c for c in filt if c not in '()')
            fs = filt.split(' AND ')
            for f in fs:
     #           print(filters)
                # 修复原版可能因字符串包含空格而分割错误的 Bug
                # 例如 name = 'Alice Bob' 会被 split(' ') 拆坏
                # 这里我们假设操作符前后有空格，限制拆分次数为 2
                parts = f.strip().split(' ', 2)
                if len(parts) < 3:
                    print(f"Warning: Filter '{f}' does not have the expected format 'col op val'. Skipping.")
                    continue
                col = parts[0]
                op = parts[1]
                num_str = parts[2].strip("'").strip('"') # 去除 SQL 字符串两端的引号

                if '.' in col:
                    column = col
                else:
                    column = str(alias) + '.' + col if alias else col
    #            print(f)

                col_id = self.col2idx.get(column, self.col2idx.get('NA', 0))
                op_id = self.op2idx.get(op, self.op2idx.get('NA', 0))
                
                res['colId'].append(col_id)
                res['opId'].append(op_id)
                res['val'].append(self.normalize_val(column, num_str))
        if len(res['colId']) == 0:
            return {'colId':[self.col2idx['NA']], 'opId': [self.op2idx['NA']], 'val': [0.0]} 
        return res
    
    def encode_join(self, join):
        if join is None:
            return 0
        if join not in self.join2idx:
            self.join2idx[join] = len(self.join2idx)
            self.idx2join[self.join2idx[join]] = join
            # return self.join2idx.get('NA', 0)
        return self.join2idx[join]
    
    def encode_table(self, table):
        if table not in self.table2idx:
            self.table2idx[table] = len(self.table2idx)
            self.idx2table[self.table2idx[table]] = table
        return self.table2idx[table]

    def encode_type(self, nodeType):
        if nodeType not in self.type2idx:
            self.type2idx[nodeType] = len(self.type2idx)
            self.idx2type[self.type2idx[nodeType]] = nodeType
        return self.type2idx[nodeType]


class TreeNode:
    def __init__(self, nodeType, typeId, filt, card, join, join_str, filterDict):
        self.nodeType = nodeType
        self.typeId = typeId
        self.filter = filt
        
        self.table = 'NA'
        self.table_id = 0
        self.query_id = None ## so that sample bitmap can recognise
        
        self.join = join
        self.join_str = join_str
        self.card = card #'Actual Rows'
        self.children = []
        self.rounds = 0
        
        self.filterDict = filterDict
        
        self.parent = None
        
        self.feature = None
        
    def addChild(self,treeNode):
        self.children.append(treeNode)
    
    def __str__(self):
#        return TreeNode.print_nested(self)
        return '{} with {}, {}, {} children'.format(self.nodeType, self.filter, self.join_str, len(self.children))

    def __repr__(self):
        return self.__str__()
    
    @staticmethod
    def print_nested(node, indent = 0): 
        print('--'*indent+ '{} with {} and {}, {} childs'.format(node.nodeType, node.filter, node.join_str, len(node.children)))
        for k in node.children: 
            TreeNode.print_nested(k, indent+1)
        
