import os
import sys
import json
import random
from collections import defaultdict
import gzip
import pandas as pd
import numpy as np

random.seed(1230)

users = defaultdict(list)
item_count = defaultdict(int)
filter_size = 5

if len(sys.argv) > 1:
    name = sys.argv[1]
if len(sys.argv) > 2:
    filter_size = int(sys.argv[2])
def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)
def get_df(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')
def read_from_amazon(df):
    # 遍历df
    for i, row in df.iterrows():
        uid = row['reviewerID']
        iid = row['asin']
        ts = float(row['unixReviewTime'])
        item_count[iid] += 1
        users[uid].append((iid, ts))

def export_data(name, user_list):
    total_data = 0
    with open(name, 'w') as f:
        for user in user_list:
            if user not in user_map:
                continue
            item_list = users[user]
            item_list.sort(key=lambda x:x[1])
            index = 0
            for item, timestamp in item_list:
                if item in item_map:
                    f.write('%d,%d,%d,%d\n' % (user_map[user], item_map[item], index, timestamp))
                    index += 1
                    total_data += 1
    return total_data
def related_filter(related_dict):
    out_dict = dict()
    if related_dict is not np.nan:
        for r in related_dict:
            out_dict[r] = list(all_items & set(related_dict[r]))
    return out_dict

def find_value(dict, key):
    if key in dict:
        return dict[key]
    else:
        return []


if __name__ == '__main__':
    DATASET = "Grocery"
    META_FILE = DATASET+'/meta_{}.json.gz'.format(DATASET)
    DATA_FILE = DATASET+'/reviews_{}_5.json.gz'.format(DATASET)
    filter_size = 5
    data_df = get_df(DATA_FILE)[["reviewerID", "asin", "unixReviewTime"]]
    read_from_amazon(data_df)

    items = list(item_count.items())
    items.sort(key=lambda x: x[1], reverse=True)

    item_total = 0
    print("Use core", filter_size)
    for index, (iid, num) in enumerate(items):
        if num >= filter_size:
            item_total = index + 1
        else:
            break

    item_map = dict(zip([items[i][0] for i in range(item_total)], list(range(1, item_total + 1))))

    user_ids = list(users.keys())
    filter_user_ids = []
    filter_seq_count = 0
    for user in user_ids:
        item_list = users[user]
        index = 0
        for item, timestamp in item_list:
            if item in item_map:
                index += 1
        if index >= filter_size:
            filter_user_ids.append(user)
            filter_seq_count += index
    user_ids = filter_user_ids

    num_users = len(user_ids)
    user_map = dict(zip(user_ids, list(range(num_users))))
    split_1 = int(num_users * 0.8)
    split_2 = int(num_users * 0.9)
    train_users = user_ids[:split_1]
    valid_users = user_ids[split_1:split_2]
    test_users = user_ids[split_2:]

    user_map_df = pd.DataFrame.from_dict(user_map, orient='index')
    user_map_df.to_csv(DATASET + '_user_map.csv',header=None)
    item_map_df = pd.DataFrame.from_dict(item_map, orient='index')
    item_map_df.to_csv(DATASET + '_item_map.csv',header=None)

    total_train = export_data(DATASET+'/'+DATASET + '_train.txt', train_users)
    total_valid = export_data(DATASET+'/'+DATASET + '_valid.txt', valid_users)
    total_test = export_data(DATASET+'/'+DATASET + '_test.txt', test_users)
    print('avg items: ', filter_seq_count/len(user_ids))
    print('total items: ', item_total)
    print('total behaviors: ', total_train + total_valid + total_test)
    print('total users: ', len(filter_user_ids))

    #加载meta
    meta_df = get_df(META_FILE)[["asin", "title", "categories", "related"]]


    useful_meta_df = meta_df[meta_df['asin'].isin(item_map.keys())]
    all_items = set(useful_meta_df['asin'].values.tolist())
    useful_meta_df['related'] = useful_meta_df['related'].apply(related_filter)

    # 处理categories
    l2_cate_lst = list()
    for cate_lst in useful_meta_df['categories']:
        l2_cate_lst.append(cate_lst[0][2] if len(cate_lst[0]) > 2 else np.nan)
    useful_meta_df['l2_category'] = l2_cate_lst
    l2_cates = sorted(useful_meta_df['l2_category'].dropna().unique())
    l2_dict = dict(zip(l2_cates, range(1, len(l2_cates) + 1)))
    l2_dict_df = pd.DataFrame(list(l2_dict.items()), columns=['Category', 'ID'])
    l2_dict_df.to_csv(DATASET + '/' + DATASET + '_l2_cate_name_dict.csv', index=False)
    useful_meta_df['l2_category'] = useful_meta_df['l2_category'].apply(lambda x: l2_dict[x] if x == x else 0)




    item_meta_data = dict()

    for idx in range(len(useful_meta_df)):
        info = useful_meta_df.iloc[idx]['related']
        item_meta_data[idx] = {
            'item_id': find_value(item_map, useful_meta_df.iloc[idx]['asin']),
            'i_name': useful_meta_df.iloc[idx]['title'],
            'i_category': useful_meta_df.iloc[idx]['l2_category'],
            'r_substitute': list(map(lambda x: find_value(item_map, x), info['also_viewed'])) if 'also_viewed' in info else [],
            'r_buy_together': list(
                map(lambda x: find_value(item_map, x), info['buy_after_viewing'])) if 'buy_after_viewing' in info else [],
            'r_complement': list(map(lambda x: find_value(item_map, x), info['also_bought'])) if 'also_bought' in info else [],
        }
    item_meta_data_df = pd.DataFrame.from_dict(item_meta_data, orient='index')
    item_meta_data_df.to_csv(DATASET+'/'+DATASET + '_item_meta_data.csv', header=True)
    item_meta_data_df[['item_id', 'i_name']].sort_values(by='item_id').to_csv(DATASET+'/'+DATASET + '_item_name.csv', index=False)
    item_meta_data_df[['item_id', 'i_category']].sort_values(by='item_id').to_csv(
        DATASET + '/' + DATASET + '_item_cate.csv', index=False)

