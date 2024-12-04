from collections import defaultdict
import math
import sys
import time
import faiss
import numpy as np
import torch
import torch.nn as nn
import os
import signal
from utils import *
import datetime
from tqdm import tqdm

error_flag = {'sig':0}

def sig_handler(signum, frame):
    error_flag['sig'] = signum
    print("segfault core", signum)

signal.signal(signal.SIGSEGV, sig_handler)

torch.set_printoptions(
    precision=2,
    threshold=np.inf,
    edgeitems=3,
    linewidth=200,
    profile=None,
    sci_mode=False
)

def train(opt, train_file, valid_file, test_file, test_iter, item_count, exp):
    exp_name = get_exp_name(opt, exp=exp)
    best_model_path = "best_model/" + exp_name + '/'

    train_data = get_DataLoader(train_file, opt, train_flag=1)
    valid_data = get_DataLoader(valid_file, opt, train_flag=0)
    test_data = get_DataLoader(test_file, opt, train_flag=0)

    train_text_path = 'data/' + opt.dataset + '/Prompts/typical_texts_emb_train_10_2.xlsx'
    train_cluster_path = 'data/' + opt.dataset + '/Prompts/clusters_with_centers_all_train_10_2.csv'
    tra_cluster_dict = load_cluster(train_cluster_path)
    train_text_dict = load_inference(train_text_path)

    model = get_model(opt, item_count)
    model = trans_to_cuda(model)
    model.set_sampler(opt)
    loss_fn = nn.CrossEntropyLoss()
    aux_loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    trials = 0
    start_time = time.time()
    print('start training: ', datetime.datetime.now())
    model.loss_fct = loss_fn
    try:
        total_loss, total_loss_1, total_loss_2, total_loss_3, total_loss_4, total_loss_5 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        iter = 0
        best_metric = 0
        # scheduler.step()
        for i, (users, targets, items, mask, times) in enumerate(train_data):
            model.train()
            iter += 1
            labels = targets
            optimizer.zero_grad()
            pos_items = trans_to_cuda(torch.Tensor(targets).long())
            targets = trans_to_cuda(torch.Tensor(targets).long())
            items = trans_to_cuda(torch.Tensor(items).long())
            mask = trans_to_cuda(torch.Tensor(mask).long())

            interests, atten, readout, selection = None, None, None, None
            time_mat, adj_mat = times
            times_tensor = (trans_to_cuda(torch.Tensor(time_mat).float()), trans_to_cuda(torch.Tensor(adj_mat).float()))
            if opt.model_type in ['ComiRec-SA', "REMI"]:
                interests, scores, text_scores, atten, readout, selection, align_loss = model(items, pos_items, mask, times_tensor, users,
                                                              train_text_dict, tra_cluster_dict)
            if opt.model_type == 'ComiRec-DR':
                interests, scores, readout = model(items, pos_items, mask, times_tensor)
            if opt.model_type == 'MIND':
                interests, scores, text_scores, readout, selection, align_loss = model(items, pos_items, mask, times_tensor, users,
                                                              train_text_dict, tra_cluster_dict)
            if opt.model_type == 'GRU4Rec':
                readout, scores = model(items, pos_items, mask, times_tensor)
            if opt.model_type in ['SASRec', 'Bert4Rec', 'DNN']:
                interests, scores, text_scores, readout, selection, align_loss = model(items, pos_items, mask, times_tensor, users,
                                                              train_text_dict, tra_cluster_dict)

            loss = model.calculate_sampled_loss(readout, pos_items, selection,
                                        interests) if model.is_sampler else model.calculate_full_loss(loss_fn, scores,
                                                                                                       targets,
                                                                                                      interests)

            if opt.model_type in ["SASRec", "Bert4Rec", "MIND"]:
                aux_loss = aux_loss_fn(text_scores, targets)
                loss = loss + align_loss * 0.1 + aux_loss * 0.1
            if opt.model_type == "REMI":
                aux_loss = aux_loss_fn(text_scores, targets)
                loss = loss + align_loss * 0.1 + aux_loss * 0.1
                loss += opt.rlambda * model.calculate_atten_loss(atten)


            loss.backward()
            optimizer.step()

            total_loss += loss

            if iter % test_iter == 0:
                model.eval()
                metrics = evaluate(model, valid_data, opt, opt.topN)

                log_str = 'iter: %d, train loss: %.4f' % (iter, total_loss / test_iter)
                if metrics != {}:
                    log_str += ', ' + ', '.join(['valid ' + key + ': %.6f' % value for key, value in metrics.items()])
                print(exp_name)
                print(log_str)

                if 'recall' in metrics:
                    recall = metrics['recall']
                    if recall > best_metric:
                        best_metric = recall
                        save_model(model, best_model_path)
                        trials = 0
                    else:
                        trials += 1
                        if trials > opt.patience:
                            print("early stopping!")
                            break

                total_loss = 0.0
                test_time = time.time()
                print("time interval: %.4f min" % ((test_time - start_time) / 60.0))
                sys.stdout.flush()

            if iter >= opt.max_iter * 1000:
                break
    except KeyboardInterrupt:
        print('-' * 99)
        print('Exiting from training early')

    load_model(model, best_model_path)
    model.eval()

    metrics = evaluate(model, valid_data, opt, opt.topN)
    print(', '.join(['Valid ' + key + ': %.6f' % value for key, value in metrics.items()]))

    print("Test result:")
    metrics = evaluate(model, test_data, opt, topN=20)
    for key, value in metrics.items():
        print('test ' + key + '@20' + '=%.6f' % value)

    metrics = evaluate(model, test_data, opt, topN=50)
    for key, value in metrics.items():
        print('test ' + key + '@50' + '=%.6f' % value)


def test(opt, test_file, cate_file, item_count, coef=None, exp='e1'):
    exp_name = get_exp_name(opt, exp, save=False)
    best_model_path = "best_model/" + exp_name + '/'
    model = get_model(opt, item_count)
    load_model(model, best_model_path)
    model = trans_to_cuda(model)
    model.eval()

    test_data = get_DataLoader(test_file, opt, train_flag=0)
    metrics = evaluate(model, test_data, opt, topN=20)
    for key, value in metrics.items():
        print('test ' + key + '@20' + '=%.6f' % value)

    metrics = evaluate(model, test_data, opt, topN=50)
    for key, value in metrics.items():
        print('test ' + key + '@50' + '=%.6f' % value)



def evaluate(model, test_data, opt, topN=20, text_dict=None, cluster_dict=None, coef=None, item_cate_map=None):
    topN = topN
    gpu_indexs = [None]
    for i in range(1000):
        try:
            item_embs = model.output_items().cpu().detach().numpy()
            res = faiss.StandardGpuResources()
            flat_config = faiss.GpuIndexFlatConfig()
            flat_config.device = opt.gpu
            gpu_indexs[0] = faiss.GpuIndexFlatIP(res, opt.hidden_size, flat_config)
            gpu_indexs[0].add(item_embs)
            if error_flag['sig'] == 0:
                break
            else:
                print("core received", error_flag['sig'])
                error_flag['sig'] = 0
        except Exception as e:
            print("error received", e)
        print("Faiss re-try", i)
        time.sleep(5)

    total = 0
    total_recall = 0.0
    total_ndcg = 0.0
    total_hitrate = 0
    total_diversity = 0.0

    for _, (users, targets, items, mask, times) in enumerate(test_data):
        time_mat, adj_mat = times
        items = trans_to_cuda(torch.Tensor(items).long())
        mask = trans_to_cuda(torch.Tensor(mask).long())
        time_tensor = (trans_to_cuda(torch.Tensor(time_mat).float()), trans_to_cuda(torch.Tensor(adj_mat).float()))

        user_embs, text_score = model(items, None, mask, time_tensor, users, text_dict, cluster_dict, train=False)
        user_embs = user_embs.cpu().detach().numpy()
        gpu_index = gpu_indexs[0]
        if len(user_embs.shape) == 2:
            D, I = gpu_index.search(user_embs, topN)
            for i, iid_list in enumerate(targets):
                recall = 0
                dcg = 0.0
                item_list = set(I[i])
                for no, iid in enumerate(item_list):
                    if iid in iid_list:
                        recall += 1
                        dcg += 1.0 / math.log(no+2, 2)
                idcg = 0.0
                for no in range(recall):
                    idcg += 1.0 / math.log(no+2, 2)
                total_recall += recall * 1.0 / len(iid_list)
                if recall > 0:
                    total_ndcg += dcg / idcg
                    total_hitrate += 1
                if coef is not None:
                    total_diversity += compute_diversity(I[i], item_cate_map)

        else:
            ni = user_embs.shape[1]
            user_embs = np.reshape(user_embs,[-1, user_embs.shape[-1]])
            D, I = gpu_index.search(user_embs, topN)
            for i, iid_list in enumerate(targets):
                recall = 0
                dcg = 0.0
                item_list_set = set()
                if coef is None:
                    item_list = list(zip(np.reshape(I[i*ni:(i+1)*ni], -1), np.reshape(D[i*ni:(i+1)*ni], -1)))
                    item_list.sort(key=lambda x:x[1], reverse=True)
                    for j in range(len(item_list)):
                        if item_list[j][0] not in item_list_set and item_list[j][0] != 0:
                            item_list_set.add(item_list[j][0])
                            if len(item_list_set) >= topN:
                                break
                else:
                    coef = float(coef)
                    origin_item_list = list(
                        zip(np.reshape(I[i * ni:(i + 1) * ni], -1), np.reshape(D[i * ni:(i + 1) * ni], -1)))
                    origin_item_list.sort(key=lambda x: x[1], reverse=True)
                    item_list = []
                    tmp_item_set = set()
                    for (x, y) in origin_item_list:
                        if x not in tmp_item_set and x in item_cate_map:
                            item_list.append((x, y, item_cate_map[x]))
                            tmp_item_set.add(x)
                    cate_dict = defaultdict(int)
                    for j in range(topN):
                        max_index = 0
                        max_score = item_list[0][1] - coef * cate_dict[item_list[0][2]]
                        for k in range(1, len(item_list)):
                            if item_list[k][1] - coef * cate_dict[item_list[k][2]] > max_score:
                                max_index = k
                                max_score = item_list[k][1] - coef * cate_dict[item_list[k][2]]
                            elif item_list[k][1] < max_score:
                                break
                        item_list_set.add(item_list[max_index][0])
                        cate_dict[item_list[max_index][2]] += 1
                        item_list.pop(max_index)

                for no, iid in enumerate(item_list_set):
                    if iid in iid_list:
                        recall += 1
                        dcg += 1.0 / math.log(no + 2, 2)
                idcg = 0.0
                for no in range(recall):
                    idcg += 1.0 / math.log(no + 2, 2)
                total_recall += recall * 1.0 / len(iid_list)
                if recall > 0:
                    total_ndcg += dcg / idcg
                    total_hitrate += 1
                if coef is not None:
                    total_diversity += compute_diversity(list(item_list_set), item_cate_map)

        total += len(targets)

    recall = total_recall / total
    ndcg = total_ndcg / total
    hitrate = total_hitrate * 1.0 / total
    if coef is None:
        return {'recall': recall, 'ndcg': ndcg, 'hitrate': hitrate}
    diversity = total_diversity * 1.0 / total
    return {'recall': recall, 'ndcg': ndcg, 'hitrate': hitrate, 'diversity': diversity}


