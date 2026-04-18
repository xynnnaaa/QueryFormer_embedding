import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .dataset import PlanTreeDataset
from .database_util import collator, get_job_table_sample, get_join_embedding
import os
import time
import torch
from scipy.stats import pearsonr

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def print_qerror(preds_unnorm, labels_unnorm, prints=False):
    qerror = []
    for i in range(len(preds_unnorm)):
        pred = max(preds_unnorm[i], 1e-6)
        label = max(float(labels_unnorm[i]), 1e-6)

        if pred > label:
            qerror.append(pred / label)
        else:
            qerror.append(label / pred)

        # if preds_unnorm[i] > float(labels_unnorm[i]):
        #     qerror.append(preds_unnorm[i] / float(labels_unnorm[i]))
        # else:
        #     qerror.append(float(labels_unnorm[i]) / float(preds_unnorm[i]))

    e_50, e_90 = np.median(qerror), np.percentile(qerror,90) 
    e_95 = np.percentile(qerror, 95)
    e_max = np.max(qerror)   
    e_mean = np.mean(qerror)

    if prints:
        print('QError 50th: {:.4f}, 90th: {:.4f}, 95th: {:.4f}, Mean: {:.4f}, Max: {:.4f}'.format(
            e_50, e_90, e_95, e_mean, e_max))

    res = {
        'q_median' : e_50,
        'q_90' : e_90,
        'q_95' : e_95, # 【新增】加入返回值
        'q_mean' : e_mean,
        'q_max' : e_max
    }

    return res

def get_corr(ps, ls): # unnormalised
    ps = np.array(ps)
    ls = np.array(ls)
    corr, _ = pearsonr(np.log(ps + 1e-6), np.log(ls + 1e-6))
    
    return corr


def eval_workload(workload, methods, use_single_embedding=False, embedding_file=None, sample_dim=1000):

    get_table_sample = methods['get_sample']

    data_path = methods.get('data_path', './data/genome/')

    workload_file_name = data_path + 'query/' + workload
    table_sample = get_table_sample(workload_file_name, use_single_embedding=use_single_embedding, embedding_file=embedding_file)
    plan_df = pd.read_csv(data_path + 'query/{}_plan.csv'.format(workload))
    workload_csv = pd.read_csv(data_path + 'query/{}.csv'.format(workload),sep='#',header=None)
    workload_csv.columns = ['table','join','predicate','cardinality']

    use_join_embedding = methods.get('use_join_embedding', False)
    join_embedding_dim = methods.get('join_embedding_dim', 768)
    test_join_embs = get_join_embedding(methods.get('test_join_embedding_file', None), use_join_embedding, join_embedding_dim)

    ds = PlanTreeDataset(plan_df, workload_csv, \
        methods['encoding'], methods['hist_file'], methods['card_norm'], \
        methods['cost_norm'], 'card', table_sample,
        sample_dim=methods['sample_dim'], max_filters=methods['max_filters'],
        use_join_embedding=use_join_embedding, join_embeddings=test_join_embs, join_dim=join_embedding_dim
        )

    eval_score, _, unnorm_preds = evaluate(methods['model'], ds, methods['bs'], methods['card_norm'], methods['device'],True)

    # 保存测试集结果
    save_path = methods.get('newpath', './results/')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    prediction_file = os.path.join(save_path, f'prediction.csv')

    results_df = pd.DataFrame({
        'predicted': unnorm_preds,
        'actual': ds.gts
    })
    results_df.to_csv(prediction_file, index=False)
    print(f"Predictions saved to {prediction_file}")

    return eval_score, ds


def evaluate(model, ds, bs, norm, device, prints=False):
    # print(ds.gts[:10])
    model.eval()
    cost_predss = np.empty(0)

    with torch.no_grad():
        for i in range(0, len(ds), bs):
            batch, batch_labels = collator(list(zip(*[ds[j] for j in range(i,min(i+bs, len(ds)) ) ])))

            batch = batch.to(device)

            cost_preds, _ = model(batch)
            cost_preds = cost_preds.squeeze()

            cost_predss = np.append(cost_predss, cost_preds.cpu().detach().numpy())

    unnorm_preds = norm.unnormalize_labels(cost_predss)
    scores = print_qerror(unnorm_preds, ds.gts, prints)
    corr = get_corr(unnorm_preds, ds.gts)
    if prints:
        print('Corr: ',corr)
    return scores, corr, unnorm_preds

def train(model, train_ds, val_ds, test_ds, crit, \
    norm, args, optimizer=None, scheduler=None):
    
    to_pred, bs, device, epochs, clip_size = \
        args.to_predict, args.bs, args.device, args.epochs, args.clip_size
    lr = args.lr

    if not optimizer:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if not scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, 0.7)


    t0 = time.time()
    rng = np.random.default_rng()

    # best_prev = 999999
    # best_model_path = None

    # all_checkpoints = {}
    best_checkpoints = {}
    checkpoint_path = os.path.join(args.newpath, 'best_models.pt')

    # 用于追踪 4 个维度最佳状态的字典
    best_stats = {
        'val_mean': {'epoch': -1, 'val': float('inf'), 'state_dict': None},
        'val_median': {'epoch': -1, 'val': float('inf'), 'state_dict': None},
        'test_mean': {'epoch': -1, 'val': float('inf'), 'state_dict': None},
        'test_median': {'epoch': -1, 'val': float('inf'), 'state_dict': None}
    }


    for epoch in range(epochs):
        losses = 0
        predss = np.empty(0)

        model.train()

        train_idxs = rng.permutation(len(train_ds))

        labelss = np.array(train_ds.gts)[train_idxs]


        for idxs in chunks(train_idxs, bs):
            optimizer.zero_grad()

            batch, batch_labels = collator(list(zip(*[train_ds[j] for j in idxs])))
            
            l, r = zip(*(batch_labels))

            batch_label = torch.FloatTensor(l).to(device)
            batch = batch.to(device)

            cost_preds, _ = model(batch)
            cost_preds = cost_preds.squeeze()

            loss = crit(cost_preds, batch_label)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_size)

            optimizer.step()
            # # SQ: added the following 3 lines to fix the out of memory issue
            # del batch
            # del batch_labels
            # torch.cuda.empty_cache()

            losses += loss.item() * batch_label.size(0)
            predss = np.append(predss, cost_preds.detach().cpu().numpy())

            # SQ: added the following 3 lines to fix the out of memory issue
            del batch
            del batch_labels
            torch.cuda.empty_cache()

        # 每个epoch结束评估验证集和测试集

        val_scores, _, _ = evaluate(model, val_ds, bs, norm, device, False)
        test_scores, _, _ = evaluate(model, test_ds, bs, norm, device, False)

        cur_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        updated = False

        if val_scores['q_mean'] < best_stats['val_mean']['val']:
            best_stats['val_mean'] = {'epoch': epoch, 'val': val_scores['q_mean'], 'state_dict': cur_state}
            updated = True
        if val_scores['q_median'] < best_stats['val_median']['val']:
            best_stats['val_median'] = {'epoch': epoch, 'val': val_scores['q_median'], 'state_dict': cur_state}
            updated = True
        if test_scores['q_mean'] < best_stats['test_mean']['val']:
            best_stats['test_mean'] = {'epoch': epoch, 'val': test_scores['q_mean'], 'state_dict': cur_state}
            updated = True
        if test_scores['q_median'] < best_stats['test_median']['val']:
            best_stats['test_median'] = {'epoch': epoch, 'val': test_scores['q_median'], 'state_dict': cur_state}
            updated = True

        print(f"Epoch {epoch} | Train Avg Loss: {losses/len(train_ds):.6f} | Time: {time.time()-t0:.1f}s")
        print(f"  => [Val]  Mean: {val_scores['q_mean']:7.4f}, Median: {val_scores['q_median']:7.4f}")
        print(f"  => [Test] Mean: {test_scores['q_mean']:7.4f}, Median: {test_scores['q_median']:7.4f}")

        if updated or (epoch == epochs - 1):
            save_dict = {
                'val_mean': best_stats['val_mean']['state_dict'],
                'val_median': best_stats['val_median']['state_dict'],
                'test_mean': best_stats['test_mean']['state_dict'],
                'test_median': best_stats['test_median']['state_dict']
            }

            meta_info = {
                'val_mean_epoch': best_stats['val_mean']['epoch'],
                'val_median_epoch': best_stats['val_median']['epoch'],
                'test_mean_epoch': best_stats['test_mean']['epoch'],
                'test_median_epoch': best_stats['test_median']['epoch'],
                'current_epoch': epoch
            }
            torch.save({'models': save_dict, 'meta': meta_info}, checkpoint_path)
            if updated:
                print(f"  => Best model updated at epoch {epoch}, checkpoint saved")

        # if epoch > 40:
        #     test_scores, corrs, _ = evaluate(model, val_ds, bs, norm, device, False)

        #     if test_scores['q_mean'] < best_prev: ## mean mse
        #         best_model_path = logging(args, epoch, test_scores, filename = 'log.txt', save_model = True, model = model)
        #         best_prev = test_scores['q_mean']
        #         print(f"--> New best model saved at epoch {epoch} with q_mean: {test_scores['q_mean']:.4f} and corr: {corrs:.4f}")

        # if epoch % 10 == 0:
        #     print('Epoch: {}  Avg Loss: {}, Time: {}'.format(epoch,losses/len(train_ds), time.time()-t0))
        #     train_scores = print_qerror(norm.unnormalize_labels(predss),labelss, True)

        scheduler.step()   

    print("\n" + "="*50)
    print("                 TRAINING SUMMARY                 ")
    print("="*50)
    eval_targets = [
        ("Best Val Mean Q-Error", best_stats['val_mean']),
        ("Best Val Median Q-Error", best_stats['val_median']),
        ("Best Test Mean Q-Error", best_stats['test_mean']),
        ("Best Test Median Q-Error", best_stats['test_median'])
    ]

    evaluated_epochs = {}

    for target_name, best_info in eval_targets:
        best_epoch = best_info['epoch']
        print(f"\n--- Evaluating Model from [ {target_name} ] (Epoch {best_epoch}) ---")
        if best_epoch == -1:
            print(f"  -> Warning: No valid best epoch found for {target_name}")
            continue
        else:
            target_state_dict = best_info['state_dict']
            model.load_state_dict(target_state_dict)
            model = model.to(device)
            scores, corrs, _ = evaluate(model, test_ds, bs, norm, device, prints=False)
            evaluated_epochs[best_epoch] = scores
        print(f"  -> Test Set Q-Error | 50th: {scores['q_median']:.4f} | 90th: {scores['q_90']:.4f} | "
              f"95th: {scores['q_95']:.4f} | Mean: {scores['q_mean']:.4f} | Max: {scores['q_max']:.4f}")
        
    print("\n" + "="*80)
    print(f"All epoch states saved to: {checkpoint_path}\n")

    return model


def logging(args, epoch, qscores, filename = None, save_model = False, model = None):
    arg_keys = [attr for attr in dir(args) if not attr.startswith('__')]
    arg_vals = [getattr(args, attr) for attr in arg_keys]
    
    res = dict(zip(arg_keys, arg_vals))
    model_checkpoint = str(hash(tuple(arg_vals))) + '.pt'

    res['epoch'] = epoch
    res['model'] = model_checkpoint 


    res = {**res, **qscores}

    filename = args.newpath + '/' + filename
    model_checkpoint = args.newpath + '/' + model_checkpoint

    if filename is not None:
        if os.path.isfile(filename):
            df = pd.read_csv(filename)
            res_df = pd.DataFrame([res])
            df = pd.concat([df, res_df], ignore_index=True)
            df.to_csv(filename, index=False)
        else:
            df = pd.DataFrame(res, index=[0])
            df.to_csv(filename, index=False)
    if save_model:
        torch.save({
            'model': model.state_dict(),
            'args' : args
        }, model_checkpoint)
    
    return res['model']  