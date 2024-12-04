import torch
import numpy as np
import random
import os
import pandas as pd
import shutil

from torch.utils.data import DataLoader
from Models.MIND import MIND
from Models.GRU4Rec import GRU4Rec
from Models.SASRec import SASRec
from Models.Bert4Rec import BERT4Rec
from Models.ComiRec import ComiRec_SA
from Models.DNN import DNN
from Models.REMI import REMI
from tqdm import tqdm

def get_model(opt, item_count, routing_times=3):
    add_pos = True
    if opt:
        add_pos = opt.add_pos == 1
    model_type = opt.model_type
    dataset = opt.dataset
    if model_type == 'MIND':
        relu_layer = True if dataset == 'book' else False
        model = trans_to_cuda(MIND(opt, item_count, opt.hidden_size, opt.batch_size, opt.interest_num, opt.seq_len, routing_times=routing_times, relu_layer=relu_layer))
    elif model_type == 'MIND_original':
        relu_layer = True if dataset == 'book' else False
        model = trans_to_cuda(MIND_original(item_count, opt.hidden_size, opt.batch_size, opt.interest_num, opt.seq_len, routing_times=routing_times,
                     relu_layer=relu_layer))
    elif model_type == 'GRU4Rec':
        model = GRU4Rec(item_count, opt.hidden_size, opt.batch_size, opt.seq_len, num_layers=opt.layers, dropout=opt.dropout)
    elif model_type == 'DNN':
        model = DNN(opt, item_count, opt.hidden_size, opt.batch_size, opt.seq_len)
    elif model_type == 'SASRec':
        model = SASRec(opt, item_count, opt.hidden_size, opt.batch_size, opt.seq_len, dropout=opt.dropout)
    elif model_type == 'Bert4Rec':
        model = BERT4Rec(opt, item_count, opt.hidden_size, opt.batch_size, opt.seq_len, dropout=opt.dropout)
    elif model_type in ['ComiRec-SA']:
        model = ComiRec_SA(opt, item_count, opt.hidden_size, opt.batch_size, opt.interest_num, opt.seq_len, add_pos=add_pos, args = opt)
    elif model_type == "REMI":
        model = REMI(opt, item_count, opt.hidden_size, opt.batch_size, opt.interest_num, opt.seq_len, add_pos=add_pos, beta=opt.rbeta)
    else:
        print ("Invalid model_type : %s", model_type)
        return
    model.name = model_type
    return model
def get_DataLoader(source, opt, train_flag=1):
    dataIterator = DataIterator(source, opt, train_flag)
    return DataLoader(dataIterator, batch_size=None, batch_sampler=None)

class DataIterator(torch.utils.data.IterableDataset):
    def __init__(self, source, opt, train_flag=1, time_span=128):
        print("Using time span", time_span)
        self.read(source)
        self.users = list(self.users)
        self.len = len(self.users)
        self.time_span = time_span
        self.batch_size = opt.batch_size
        self.eval_batch_size = opt.batch_size
        self.train_flag = train_flag
        self.seq_len = opt.seq_len
        self.index = 0
        print("total user:", len(self.users))
        print("total items:", len(self.items))

    def __iter__(self):
        return self

    def read(self, source):
        self.graph = {}
        self.time_graph = {}
        self.users = set()
        self.items = set()
        self.times = set()
        with open(source, 'r') as f:
            for line in f:
                conts = line.strip().split(',')
                user_id = int(conts[0])
                item_id = int(conts[1])
                if len(conts) == 3:
                    time_stamp = int(conts[2])
                else:
                    idx = int(conts[2])
                    time_stamp = int(conts[3])
                self.users.add(user_id)
                self.items.add(item_id)
                self.times.add(time_stamp)
                if user_id not in self.graph:
                    self.graph[user_id] = []
                self.graph[user_id].append((item_id, time_stamp))
        for user_id, value in self.graph.items():
            value.sort(key=lambda x: x[1])
            time_list = list(map(lambda x: x[1], value))
            time_min = min(time_list)
            self.graph[user_id] = [x[0] for x in value]
            self.time_graph[user_id] = [int(round((x[1] - time_min) / 86400.0) + 1) for x in value]
        self.users = list(self.users)
        self.items = list(self.items)

    def compute_time_matrix(self, time_seq, item_num):
        time_matrix = np.zeros([self.seq_len, self.seq_len], dtype=np.int32)  # [s,s]
        for i in range(item_num):
            for j in range(item_num):
                span = abs(time_seq[i] - time_seq[j])
                if span > self.time_span:
                    time_matrix[i][j] = self.time_span
                else:
                    time_matrix[i][j] = span
        return time_matrix.tolist()


    def compute_adj_matrix(self, mask_seq, item_num):  # 序列，有效元素数量
        node_num = len(mask_seq)

        adj_matrix = np.zeros([node_num, node_num + 2], dtype=np.int32)

        adj_matrix[0][0] = 1
        adj_matrix[0][1] = 1
        adj_matrix[0][-1] = 1

        adj_matrix[item_num - 1][item_num - 1] = 1
        adj_matrix[item_num - 1][item_num] = 1
        adj_matrix[item_num - 1][-1] = 1

        for i in range(1, item_num - 1):
            adj_matrix[i][i] = 1
            adj_matrix[i][i + 1] = 1
            adj_matrix[i][-1] = 1

        if (item_num < node_num):
            for i in range(item_num, node_num):
                adj_matrix[i][0] = 1
                adj_matrix[i][1] = 1
                adj_matrix[i][-1] = 1

        return adj_matrix.tolist()

    def __next__(self):
        if self.train_flag == 1:
            user_id_list = random.sample(self.users, self.batch_size)
        else:
            total_user = len(self.users)
            if self.index >= total_user:
                self.index = 0
                raise StopIteration
            user_id_list = self.users[self.index: self.index + self.eval_batch_size]
            self.index += self.eval_batch_size

        item_id_list = []
        hist_time_list = []
        hist_item_list = []
        time_matrix_list = []
        hist_mask_list = []
        adj_matrix_list = []

        for user_id in user_id_list:
            item_list = self.graph[user_id]
            time_list = self.time_graph[user_id]
            if self.train_flag == 1:
                k = random.choice(range(4, len(item_list)))
                item_id_list.append(item_list[k])
            else:
                k = int(len(item_list) * 0.8)
                item_id_list.append(item_list[k:])
            if k >= self.seq_len:
                hist_item_list.append(item_list[k - self.seq_len: k])
                hist_mask_list.append([1.0] * self.seq_len)
                hist_time_list.append(time_list[k - self.seq_len: k])
                time_matrix_list.append(self.compute_time_matrix(time_list[k - self.seq_len: k], self.seq_len))
                adj_matrix_list.append(self.compute_adj_matrix([1.0] * self.seq_len, self.seq_len))
            else:
                hist_item_list.append(item_list[:k] + [0] * (self.seq_len - k))
                hist_mask_list.append([1.0] * k + [0.0] * (self.seq_len - k))
                hist_time_list.append(time_list[:k] + [0] * (self.seq_len - k))
                time_matrix_list.append(self.compute_time_matrix(time_list[:k] + [0] * (self.seq_len - k), k))
                adj_matrix_list.append(self.compute_adj_matrix([1.0] * k + [0.0] * (self.seq_len - k), k))
        return user_id_list, item_id_list, hist_item_list, hist_mask_list, (time_matrix_list, adj_matrix_list)


def load_inference(file_path):
    texts_df = pd.read_excel(file_path)
    texts_dict = {}
    for index, row in tqdm(texts_df.iterrows(), desc="load text inference"):
        cluster_id = row['Cluster_Center_Index']
        text_emb = row['Text_Embedding'] #str
        texts_dict[cluster_id] = text_emb
    return texts_dict


def load_cluster(file_path):
    cluster_df = pd.read_csv(file_path)
    cluster_dict = {}
    for index, row in cluster_df.iterrows():
        user_id = row['ID']
        cluster_id = row['Cluster_Center_Index']
        cluster_dict[user_id] = cluster_id
    return cluster_dict


def save_model(model, Path):
    if not os.path.exists(Path):
        os.makedirs(Path)
    torch.save(model.state_dict(), Path + 'model.pt')

def load_model(model, path):
    model.load_state_dict(torch.load(path + 'model.pt'))
    print('model loaded from %s' % path)

def load_item_name(source):
    item_relation = {}
    df = pd.read_csv(source)
    for index, row in df.iterrows():
        key = row['item_id']
        value = row['i_name']
        item_relation[key] = value

def load_item_text_emb(file_path):
    df = pd.read_csv(file_path)
    item_text_emb_dict = {}
    for index, row in df.iterrows():
        key = row['item_id']
        value = row['Text_Embedding']
        item_text_emb_dict[key] = value
    return item_text_emb_dict

def load_item_cate(source):
    item_cate = {}
    with open(source, 'r') as f:
        for line in f:
            conts = line.strip().split(',')
            item_id = int(conts[0])
            cate_id = int(conts[1])
            item_cate[item_id] = cate_id
    return item_cate

def compute_diversity(item_list, item_cate_map):
    n = len(item_list)
    diversity = 0.0
    for i in range(n):
        for j in range(i+1, n):
            diversity += item_cate_map[item_list[i]] != item_cate_map[item_list[j]]
    diversity /= ((n-1) * n / 2)
    return diversity

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_exp_name(opt, exp='e1', save=True):
    extr_name = exp
    para_name = '_'.join([opt.dataset, opt.model_type, 'b' + str(opt.batch_size), 'lr' + str(opt.lr), 'd' + str(opt.hidden_size),
                          'len' + str(opt.seq_len), 'in' + str(opt.interest_num)])
    exp_name = para_name + '_' + extr_name

    while os.path.exists('best_model/' + exp_name) and save:
        # flag = input('The exp name already exists. Do you want to cover? (y/n)')
        # if flag == 'y' or flag == 'Y':
        shutil.rmtree('best_model/' + exp_name)
        break
        # else:
        #     extr_name = input('Please input the experiment name: ')
        #     exp_name = para_name + '_' + extr_name
    return exp_name