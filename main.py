import argparse
import os
import time
import torch
from utils import *
from evaluate import train, test

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Office_Products', help='Beauty | Grocery | Office_Products')
parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
parser.add_argument('--hidden_size', type=int, default=64)
parser.add_argument('--seq_len', type=int, default=20, help='sequence length, Beauty 20')
parser.add_argument('--interest_num', type=int, default=4)
parser.add_argument('--model_type', type=str, default='REMI', help='DNN | GRU4Rec | MIND | REMI..') #select backbones
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning_rate')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=30, help='(k), the number of steps after which the learning rate decay')
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--gpu', type=int, default=0) # None -> cpu
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--topN', type=int, default=50)
parser.add_argument('--phrase', type=str, default='train', help='train | test')
parser.add_argument('--coef', default=None, help='diversity test')
parser.add_argument('--exp', default='e10')
parser.add_argument('--random_seed', type=int, default=2021)
parser.add_argument('--add_pos', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--layers', type=int, default=1)
parser.add_argument('--sampled_n', type=int, default=0)
parser.add_argument('--sampled_loss', type=str, default='sampled')
parser.add_argument('--sample_prob', type=int, default=0)
parser.add_argument('--max_iter', type=int, default=1000, help='(k)')
# For REMI
parser.add_argument('--rbeta', type=float, default=10)
parser.add_argument('--rlambda', type=float, default=100)

opt = parser.parse_args()
print(opt)
torch.cuda.set_device(opt.gpu)


def main():
    SEED = opt.random_seed
    setup_seed(SEED)
    item_count = 0
    test_iter = 0
    path = './data/{}/'.format(opt.dataset)
    if opt.dataset == 'Beauty':
        test_iter = 200
        item_count = 12101 + 1
    elif opt.dataset == 'Book':
        test_iter = 200
        item_count = 10001 + 1
    elif opt.dataset == 'Office_Products':
        test_iter = 200
        item_count = 2420 + 1
    elif opt.dataset == 'Grocery':
        test_iter = 200
        item_count = 8713 + 1

    train_file = path + '{}_train.txt'.format(opt.dataset)
    valid_file = path + '{}_valid.txt'.format(opt.dataset)
    test_file = path + '{}_test.txt'.format(opt.dataset)
    cate_file = path + '{}_item_cate.csv'.format(opt.dataset)
    name_file = path + '{}_item_name.csv'.format(opt.dataset)
    meta_file = path + '{}_item_meta_data.csv'.format(opt.dataset)
    prob_dic = {
        0: 'uniform',
        1: 'log'
    }
    if opt.phrase == 'train':
        train(opt, train_file, valid_file, test_file, test_iter, item_count, exp=opt.exp)
    elif opt.phrase == 'test':
        test(opt, test_file, cate_file, item_count, coef=opt.coef, exp=opt.exp)
    else:
        print('No phrase')


if __name__ == '__main__':
    main()