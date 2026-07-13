import sys
import os
import json
import torch
import pandas as pd
import numpy as np
import argparse
from scipy.stats import pearsonr

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
    pred_hid = 512     # 请确认是否与训练时一致
    ffn_dim = 256      # 请确认是否与训练时一致
    head_size = 12
    n_layers = 8
    dropout = 0.1
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# ----------------- QueryFormer 原生评估函数 -----------------
def print_qerror(preds_unnorm, labels_unnorm, prints=False):
    qerror = []
    for i in range(len(preds_unnorm)):
        pred = max(preds_unnorm[i], 1e-6)
        label = max(float(labels_unnorm[i]), 1e-6)

        if pred > label:
            qerror.append(pred / label)
        else:
            qerror.append(label / pred)

    e_50, e_90 = np.median(qerror), np.percentile(qerror,90) 
    e_80 = np.percentile(qerror, 80)
    e_95 = np.percentile(qerror, 95)
    e_99 = np.percentile(qerror, 99)
    e_max = np.max(qerror)   
    e_mean = np.mean(qerror)

    if prints:
        print('QError 50th: {:.4f}, 80th: {:.4f}, 90th: {:.4f}, 95th: {:.4f}, 99th: {:.4f}, Mean: {:.4f}, Max: {:.4f}'.format(
            e_50, e_80, e_90, e_95, e_99, e_mean, e_max))

    res = {
        'q_median' : e_50,
        'q_80' : e_80,
        'q_90' : e_90,
        'q_95' : e_95,
        'q_99' : e_99,
        'q_mean' : e_mean,
        'q_max' : e_max
    }

    return res

def get_corr(ps, ls): # unnormalised
    ps = np.array(ps)
    ls = np.array(ls)
    corr, _ = pearsonr(np.log(ps + 1e-6), np.log(ls + 1e-6))
    
    return corr
# -----------------------------------------------------------

def load_model_and_infer(train_config_file, ckpt_path, args, use_sample):
    """根据单个训练配置加载数据和 QueryFormer 模型，返回该模型的全量预测结果"""
    with open(train_config_file, 'r') as f:
        config = json.load(f)

    data_path = config['data_path']
    device = args.device
    
    # 1. 加载全局依赖数据
    hist_file = get_hist_file(data_path + 'histogram_string.csv')
    encoding_ckpt = torch.load(data_path + 'encoding.pt')
    encoding = encoding_ckpt['encoding']

    # 2. 准备 Normalizer 和 恢复词表状态
    train_plan_df = pd.read_csv(data_path + 'query/train_plan.csv')
    
    card_norm = Normalizer()
    all_cards = [json.loads(p)['Plan']['Actual Rows'] for p in train_plan_df['json']]
    card_norm.normalize_labels(all_cards, reset_min_max=True)
    cost_norm = Normalizer(-3.61192, 12.290855)

    dummy_table_sample = [{} for _ in range(len(train_plan_df))]
    
    _ = PlanTreeDataset(train_plan_df, None, encoding, hist_file, card_norm, cost_norm, 'card', dummy_table_sample, 
                        sample_dim=config['sample_dim'], max_filters=config['max_filters'],
                        use_join_embedding=0, join_embeddings=None, join_dim=config['join_embedding_dim'])

    # 3. 加载真实的测试集数据
    test_workload_file = data_path + 'query/test'
    test_table_sample = get_job_table_sample(test_workload_file, use_single_embedding=config['use_single_embedding'], embedding_file=config['test_embedding_file'])
    test_join_embs = get_join_embedding(config['test_join_embedding_file'], config['use_join_embedding'], config['join_embedding_dim'])
    
    test_plan_df = pd.read_csv(data_path + 'query/test_plan.csv')
    test_workload_csv = pd.read_csv(data_path + 'query/test.csv', sep='#', header=None)
    test_workload_csv.columns = ['table','join','predicate','cardinality']
    
    test_ds = PlanTreeDataset(test_plan_df, test_workload_csv, encoding, hist_file, card_norm, cost_norm, 'card', test_table_sample, 
                              sample_dim=config['sample_dim'], max_filters=config['max_filters'],
                              use_join_embedding=config['use_join_embedding'], join_embeddings=test_join_embs, join_dim=config['join_embedding_dim'])

    # 4. 初始化模型
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
        use_sample = use_sample
    )

    # 5. 加载权重
    print(f"Loading checkpoint from {ckpt_path}...")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Error: Checkpoint file {ckpt_path} not found!")

    checkpoint = torch.load(ckpt_path, map_location='cpu')
    target_state_dict = checkpoint['models']['val_mean']
    best_epoch = checkpoint['meta']['val_mean_epoch']
    print(f"Loaded weights corresponding to Best Validation Mean Q-Error at Epoch {best_epoch}.")

    model.load_state_dict(target_state_dict)
    model = model.to(device)

    # 6. 推理评估
    # prints=False 避免单独子模型打印，保持输出干净
    _, _, unnorm_preds = evaluate(model, test_ds, bs=args.bs, norm=card_norm, device=device, prints=False)
    
    actual_labels = test_ds.gts

    return np.array(unnorm_preds), np.array(actual_labels)

def evaluate_hybrid(eval_config_path):
    with open(eval_config_path, 'r') as f:
        eval_cfg = json.load(f)
        
    args = Args()
    seed_everything()
    
    use_sample = eval_cfg.get("use_sample", True)

    # 1. 运行 Base 模型 (无 Join Embedding 的模型)
    print("\n--- Phase 1: Running Base Model (Fallback) ---")
    base_train_cfg = eval_cfg["base_model"]["train_config_path"]
    base_ckpt = eval_cfg["base_model"]["model_path"]
    
    preds_base, labels_raw = load_model_and_infer(base_train_cfg, base_ckpt, args, use_sample)
    
    final_preds = preds_base.copy()

    # 2. 运行 Join 模型并利用 hit_status 融合
    if "join_model" in eval_cfg and "hit_status_path" in eval_cfg:
        print("\n--- Phase 2: Running Join Model & Merging ---")
        join_train_cfg = eval_cfg["join_model"]["train_config_path"]
        join_ckpt = eval_cfg["join_model"]["model_path"]
        hit_status_path = eval_cfg["hit_status_path"]
        
        preds_join, _ = load_model_and_infer(join_train_cfg, join_ckpt, args, use_sample)
        
        # 加载命中状态
        hit_status = torch.load(hit_status_path).cpu().numpy().flatten()
        if len(hit_status) != len(final_preds):
            raise ValueError(f"Hit status length ({len(hit_status)}) does not match test set length ({len(final_preds)})!")
        
        # 融合逻辑
        hit_count = 0
        for i in range(len(final_preds)):
            if hit_status[i] == 1:
                final_preds[i] = preds_join[i]
                hit_count += 1
                
        print(f"\nHybrid Merge Complete! Used Join Model for {hit_count}/{len(final_preds)} queries ({(hit_count/len(final_preds)):.2%}).")
    else:
        print("\n--- No Join Model / Hit Status configured. Evaluating Base Model only. ---")

    # 3. 计算最终混合 Q-Error (使用 QueryFormer 原生函数)
    print("\n=== Final Evaluation Results ===")
    _ = print_qerror(final_preds, labels_raw, prints=True)
    corr = get_corr(final_preds, labels_raw)
    print('Corr: ', corr)

    # 4. 保存预测结果为 CSV
    output_csv = eval_cfg.get("output_csv", "queryformer_predictions.csv")
    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else '.', exist_ok=True)
    
    df_res = pd.DataFrame({
        'predicted': final_preds,
        'actual': labels_raw
    })
    
    # 保持与原脚本一样的输出格式 (无 header)
    df_res.to_csv(output_csv, index=False, header=False)
    print(f"\nSaved detailed predictions to: {output_csv}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Hybrid QueryFormer models on the test set.")
    parser.add_argument('--eval_config', help='Path to hybrid evaluation JSON config file', required=True)
    args_parse = parser.parse_args()

    evaluate_hybrid(args_parse.eval_config)

if __name__ == "__main__":
    main()