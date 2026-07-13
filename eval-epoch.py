# eval_epoch.py

import sys
import os
import json
import torch
import pandas as pd
import numpy as np
import argparse

from model.database_util import get_hist_file, get_job_table_sample, get_join_embedding
from model.model import QueryFormer
from model.dataset import PlanTreeDataset
from model.trainer import evaluate
from model.util import Normalizer, seed_everything

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class Args:
    bs = 512
    embed_size = 64
    pred_hid = 512     # 请确认是否与训练时一致 (如 256 或 512)
    ffn_dim = 256      # 请确认是否与训练时一致 (如 256 或 1024)
    head_size = 12
    n_layers = 8
    dropout = 0.1
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained QueryFormer model on the test set.")
    parser.add_argument('config_file', help='Path to config JSON file')
    parser.add_argument('ckpt_path', help='Path to best_models.pt')
    parser.add_argument('pred_output_file', help='Output CSV file for predictions')
    parser.add_argument('--use_sample', type=lambda x: str(x).lower() == 'true', default=True, 
                        help='Use sample (bitmap/embedding) in QueryFormer (default: True)')

    args_parse = parser.parse_args()

    config_file = args_parse.config_file
    ckpt_path = args_parse.ckpt_path
    pred_output_file = args_parse.pred_output_file

    args = Args()
    device = args.device
    seed_everything()

    print("\nEvaluation Configuration:")
    print(f"Config File: {config_file}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Use Sample: {args_parse.use_sample}")

    with open(config_file, 'r') as f:
        config = json.load(f)

    data_path = config['data_path']
    
    # 1. 加载全局依赖数据
    print("Loading encodings and dataset dependencies...")
    hist_file = get_hist_file(data_path + 'histogram_string.csv')
    encoding_ckpt = torch.load(data_path + 'encoding.pt')
    encoding = encoding_ckpt['encoding']

    # 2. 准备 Normalizer 和 恢复词表状态 (非常关键)
    print("Reconstructing vocabulary from Training Plans...")
    train_plan_df = pd.read_csv(data_path + 'query/train_plan.csv')
    
    card_norm = Normalizer()
    all_cards = [json.loads(p)['Plan']['Actual Rows'] for p in train_plan_df['json']]
    card_norm.normalize_labels(all_cards, reset_min_max=True)
    cost_norm = Normalizer(-3.61192, 12.290855)

    # 构造一个极简的 dummy table sample，只为了让 dataset 跑通 traversePlan，不耗费内存去加载真实 embedding
    dummy_table_sample = [{} for _ in range(len(train_plan_df))]
    
    # 构建 train_ds，其唯一目的是让 traversePlan 补全 encoding 里的 idx 映射
    _ = PlanTreeDataset(train_plan_df, None, encoding, hist_file, card_norm, cost_norm, 'card', dummy_table_sample, 
                        sample_dim=config['sample_dim'], max_filters=config['max_filters'],
                        use_join_embedding=0, join_embeddings=None, join_dim=config['join_embedding_dim'])

    print(f"Vocabulary reconstructed. Final Join Vocab Size: {len(encoding.join2idx)}")

    # 3. 加载真实的测试集数据
    print("Loading Test Data...")
    test_workload_file = data_path + 'query/test'
    test_table_sample = get_job_table_sample(test_workload_file, use_single_embedding=config['use_single_embedding'], embedding_file=config['test_embedding_file'])
    test_join_embs = get_join_embedding(config['test_join_embedding_file'], config['use_join_embedding'], config['join_embedding_dim'])
    
    test_plan_df = pd.read_csv(data_path + 'query/test_plan.csv')
    test_workload_csv = pd.read_csv(data_path + 'query/test.csv', sep='#', header=None)
    test_workload_csv.columns = ['table','join','predicate','cardinality']
    
    test_ds = PlanTreeDataset(test_plan_df, test_workload_csv, encoding, hist_file, card_norm, cost_norm, 'card', test_table_sample, 
                              sample_dim=config['sample_dim'], max_filters=config['max_filters'],
                              use_join_embedding=config['use_join_embedding'], join_embeddings=test_join_embs, join_dim=config['join_embedding_dim'])

    # 4. 初始化模型 (参数与 train.py 严格一致)
    print("Initializing Model...")
    model = QueryFormer(
        emb_size=args.embed_size, ffn_dim=args.ffn_dim, head_size=args.head_size, dropout=args.dropout, n_layers=args.n_layers,
        use_hist=True, pred_hid=args.pred_hid,
        sample_dim=config['sample_dim'], max_filters=config['max_filters'],
        use_join_embedding=config['use_join_embedding'], join_dim=config['join_embedding_dim'],
        num_tables=len(encoding.table2idx) + 10,
        num_types=len(encoding.type2idx) + 10,
        num_joins=len(encoding.join2idx) + 10,
        num_columns=len(encoding.col2idx) + 10,
        num_ops=len(encoding.op2idx) + 10,
        use_single_embedding=config['use_single_embedding'],
        use_sample = args_parse.use_sample
    )

    # 5. 从包含所有 epoch 的文件中抽取特定的 epoch 状态
    print(f"Loading checkpoint from {ckpt_path}...")
    if not os.path.exists(ckpt_path):
        print("Error: Checkpoint file not found!")
        return

    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    # 提取在 Validation Set 上 Mean Q-Error 最好的模型
    target_state_dict = checkpoint['models']['val_mean']
    best_epoch = checkpoint['meta']['val_mean_epoch']

    print(f"Successfully extracted model weights. This model achieved the best Validation Mean Q-Error at Epoch {best_epoch}.")

    model.load_state_dict(target_state_dict)
    model = model.to(device)

    # 6. 评估
    print(f"\n--- Evaluating Model (from Epoch {best_epoch}) on Test Set ---")
    # 使用 1024 的 batch size 以加快推理速度
    scores, corrs, unnorm_preds = evaluate(model, test_ds, bs=args.bs, norm=card_norm, device=device, prints=True)

    # 7. 保存预测结果
    df_res = pd.DataFrame({
        'predicted': unnorm_preds,
        'actual': test_ds.gts
    })

    df_res.to_csv(pred_output_file, index=False, header=False)

    print(f"\nSaved detailed predictions to: {pred_output_file}")

if __name__ == "__main__":
    main()