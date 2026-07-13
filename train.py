import numpy as np
import os
import torch
import torch.nn as nn
import time
import json
import pandas as pd
from scipy.stats import pearsonr
import datetime
import sys
import argparse

from model.util import Normalizer
from model.database_util import get_hist_file, get_job_table_sample, collator, get_join_embedding
from model.model import QueryFormer
from model.database_util import Encoding
from model.dataset import PlanTreeDataset
from model.trainer import eval_workload, train
from model.util import seed_everything

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# bitmap
class Args:
    bs = 256
    lr = 0.0002
    epochs = 200
    clip_size = 50
    embed_size = 64
    pred_hid = 512
    ffn_dim = 256
    head_size = 12
    n_layers = 8
    dropout = 0.1
    sch_decay = 0.6
    device = 'cuda:0'
    # newpath = './results/full/card/'
    newpath = '/data2/QueryFormer/results/full/card/'
    to_predict = 'card'



# # single table embedding
# class Args:
#     bs = 1024
#     lr = 0.001
#     epochs = 500
#     clip_size = 50
#     embed_size = 64
#     pred_hid = 256
#     ffn_dim = 1024
#     head_size = 12
#     n_layers = 8
#     dropout = 0.1
#     sch_decay = 0.6
#     device = 'cuda:0'
#     # newpath = './results/full/card/'
#     newpath = '/data/QueryFormer/results/full/card/'
#     to_predict = 'card'

# + join embedding
# class Args:
#     bs = 1024
#     lr = 0.001
#     epochs = 500
#     clip_size = 50
#     embed_size = 64
#     pred_hid = 512
#     ffn_dim = 1024
#     head_size = 12
#     n_layers = 8
#     dropout = 0.1
#     sch_decay = 0.6
#     device = 'cuda:0'
#     # newpath = './results/full/card/'
#     newpath = '/data/QueryFormer/results/full/card/'
#     to_predict = 'card'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help='Path to config JSON file')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate (overrides Args.lr)')
    parser.add_argument('--use_sample', type=lambda x: x.lower() == 'true', default=True,
                        help='Use sample in QueryFormer (default: True)')
    parser.add_argument('--bs', type=int, default=None, help='Learning rate (overrides Args.bs)')
    parser.add_argument('--epochs', type=int, default=None, help='Epoch (overrides Args.epochs)')

    args_parse = parser.parse_args()
    config_file = args_parse.config_file

    # 初始化参数
    args = Args()
    if args_parse.lr is not None:
        args.lr = args_parse.lr
    if args_parse.bs is not None:
        args.bs = args_parse.bs
    if args_parse.epochs is not None:
        args.epochs = args_parse.epochs

    # 输出必要的配置信息
    print("Training Configuration:")
    print(f"Batch Size: {args.bs}")
    print(f"Learning Rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print(f"Clip Size: {args.clip_size}")
    print(f"Embedding Size: {args.embed_size}")
    print(f"Prediction Hidden Size: {args.pred_hid}")
    print(f"FFN Dimension: {args.ffn_dim}")
    print(f"Head Size: {args.head_size}")
    print(f"Number of Layers: {args.n_layers}")
    print(f"Dropout: {args.dropout}")
    print(f"Scheduler Decay: {args.sch_decay}")
    print(f"Device: {args.device}")
    print(f"New Path: {args.newpath}")
    print(f"To Predict: {args.to_predict}")
    print(f"Use sample: {args_parse.use_sample}")


    timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
    args.newpath = os.path.join(args.newpath, f"exp_{timestamp}")


    if not os.path.exists(args.newpath):
        os.makedirs(args.newpath)

    print(f"Results will be saved to: {args.newpath}")

    with open(config_file, 'r') as f:
        config = json.load(f)


    use_single_embedding = config.get('use_single_embedding', 0)
    use_log_count = config.get('use_log_count', 0)
    base_emb_dim = 768 # 你的纯 embedding 维度
    bitmap_dim = 1000 # 原始 Bitmap 维度
    pca_dim = 768 * 2

    has_pca_in_file = False

    if use_single_embedding in [1, 2]:
        test_emb_file = config.get('test_embedding_file', None)
        if test_emb_file and os.path.exists(test_emb_file):
            raw_data = torch.load(test_emb_file)
            iterator = raw_data.values() if isinstance(raw_data, dict) else raw_data
            for q_dict in iterator:
                if len(q_dict) > 0:
                    first_tensor = list(q_dict.values())[0]
                    if first_tensor.shape[-1] >= (base_emb_dim + pca_dim):
                        has_pca_in_file = True
                    break

    valid_single_emb_dim = base_emb_dim
    if use_log_count == 1:
        valid_single_emb_dim += 1
    if has_pca_in_file:
        valid_single_emb_dim += pca_dim


    # 动态计算传给 Dataset 和 Model 的总 sample_dim
    if use_single_embedding == 0:
        total_sample_dim = bitmap_dim
    elif use_single_embedding == 1:
        total_sample_dim = valid_single_emb_dim
    elif use_single_embedding == 2:
        total_sample_dim = bitmap_dim + valid_single_emb_dim
    else:
        raise ValueError("use_single_embedding must be 0, 1, or 2.")

    config['sample_dim'] = total_sample_dim
    config['has_pca'] = has_pca_in_file
    config['emb_dim'] = base_emb_dim

    # 加载数据
    data_path = config['data_path']
    hist_file = get_hist_file(data_path + 'histogram_string.csv')
    cost_norm = Normalizer(-3.61192, 12.290855)

    # 加载编码和检查点
    encoding_ckpt = torch.load(data_path + 'encoding.pt')
    encoding = encoding_ckpt['encoding']

    # checkpoint = torch.load('checkpoints/cost_model.pt', map_location='cpu')

    # 准备训练和验证数据
    print("Loading Train and Validation Dataset...")
    dfs = []
    # 可选：使用单个文件或合并多个文件
    df = pd.read_csv(data_path + '/query/train_plan.csv')
    dfs.append(df)
    full_train_df = pd.concat(dfs)

    num_queries = len(full_train_df)
    num_train = int(len(full_train_df) * 0.9)
    num_val = num_queries - num_train

    train_df = full_train_df.iloc[:num_train].reset_index(drop=True)
    val_df = full_train_df.iloc[num_train:].reset_index(drop=True)
    print(f'Total queries: {num_queries}, Train: {len(train_df)}, Val: {len(val_df)}')

    card_norm = Normalizer()
    all_cards = [json.loads(p)['Plan']['Actual Rows'] for p in full_train_df['json']]
    card_norm.normalize_labels(all_cards, reset_min_max=True)

    table_sample = get_job_table_sample(data_path + '/query/train', use_single_embedding=config['use_single_embedding'], embedding_file=config['train_embedding_file'], num_materialized_samples=1000, single_emb_dim=valid_single_emb_dim)

    train_join_embs = get_join_embedding(config['train_join_embedding_file'], config['use_join_embedding'], config['join_embedding_dim'])
    # test_join_embs = get_join_embedding(config['test_join_embedding_file'], config['use_join_embedding'], config['join_embedding_dim'])

    to_predict = 'card'
    train_ds = PlanTreeDataset(train_df, None, encoding, hist_file, card_norm, cost_norm, to_predict, table_sample, sample_dim=total_sample_dim, max_filters=config['max_filters'],
                             use_join_embedding=config['use_join_embedding'], join_embeddings=train_join_embs, join_dim=config['join_embedding_dim'])
    val_ds = PlanTreeDataset(val_df, None, encoding, hist_file, card_norm, cost_norm, to_predict, table_sample, sample_dim=total_sample_dim, max_filters=config['max_filters'],
                             use_join_embedding=config['use_join_embedding'], join_embeddings=train_join_embs, join_dim=config['join_embedding_dim'])
    
    print("Loading Test Dataset...")
    test_workload_file = data_path + 'query/test'
    test_table_sample = get_job_table_sample(test_workload_file, use_single_embedding=config['use_single_embedding'], embedding_file=config['test_embedding_file'], num_materialized_samples=1000, single_emb_dim=valid_single_emb_dim)
    test_join_embs = get_join_embedding(config['test_join_embedding_file'], config['use_join_embedding'], config['join_embedding_dim'])
    test_plan_df = pd.read_csv(data_path + 'query/test_plan.csv')
    test_workload_csv = pd.read_csv(data_path + 'query/test.csv', sep='#', header=None)
    test_workload_csv.columns = ['table','join','predicate','cardinality']
    test_ds = PlanTreeDataset(test_plan_df, test_workload_csv, encoding, hist_file, card_norm, cost_norm, to_predict, test_table_sample, 
                              sample_dim=total_sample_dim, max_filters=config['max_filters'],
                              use_join_embedding=config['use_join_embedding'], join_embeddings=test_join_embs, join_dim=config['join_embedding_dim'])

    seed_everything()
    print(f"Final Join Vocab Size: {len(encoding.join2idx)}")

    # 创建模型
    model = QueryFormer(
        emb_size=args.embed_size,
        ffn_dim=args.ffn_dim,
        head_size=args.head_size,
        dropout=args.dropout,
        n_layers=args.n_layers,
        use_sample=args_parse.use_sample,
        use_hist=True,
        pred_hid=args.pred_hid,
        sample_dim=total_sample_dim,
        max_filters=config['max_filters'],
        use_join_embedding=config['use_join_embedding'],
        join_dim=config['join_embedding_dim'],
        num_tables=len(encoding.table2idx) + 10,
        num_types=len(encoding.type2idx) + 10,
        num_joins=len(encoding.join2idx) + 10,  # Buffer
        num_columns=len(encoding.col2idx) + 10,
        num_ops=len(encoding.op2idx) + 10,
        use_single_embedding=config['use_single_embedding'],
        use_log_count=use_log_count,
        emb_dim=base_emb_dim,
        bitmap_dim=bitmap_dim,
        has_pca=has_pca_in_file
    )
    model = model.to(args.device)

    # 训练
    crit = nn.MSELoss()
    model = train(model, train_ds, val_ds, test_ds, crit, card_norm, args)

    print("\nTraining completed. All epochs saved.")

    # if best_path:
    #     full_best_path = os.path.join(args.newpath, best_path)
    #     print(f"\n>>> Loading best model for final evaluation: {full_best_path}")
    #     checkpoint = torch.load(full_best_path)
    #     model.load_state_dict(checkpoint['model'])
    #     model = model.to(args.device)
    #     model.eval()
    # else:
    #     print("\n>>> No best model found, using the last epoch model for evaluation.")

    # # 评估方法配置
    # methods = {
    #     'get_sample': get_job_table_sample,
    #     'encoding': encoding,
    #     'cost_norm': cost_norm,
    #     'card_norm': card_norm,
    #     'hist_file': hist_file,
    #     'model': model,
    #     'device': args.device,
    #     'bs': 2048,
    #     'sample_dim': config['sample_dim'],
    #     'max_filters': config['max_filters'],
    #     'newpath': args.newpath,
    #     'data_path': data_path,
    #     'use_single_embedding': config['use_single_embedding'],
    #     'embedding_file': config['test_embedding_file'],
    #     'use_join_embedding': config['use_join_embedding'],
    #     'join_embedding_dim': config['join_embedding_dim'],
    #     'test_join_embedding_file': config['test_join_embedding_file']
    #     # 'test_join_embs': test_join_embs
    # }

    # print("Evaluating on test workload...")

    # # 在工作负载上评估
    # _ = eval_workload('test', methods, use_single_embedding=config['use_single_embedding'], embedding_file=config['test_embedding_file'], sample_dim=config['sample_dim'])


if __name__ == '__main__':
    main()