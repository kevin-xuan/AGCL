from collections import defaultdict
import random
import numpy as np
import pandas as pd
import json
import pickle
import gzip
import tqdm
from datetime import datetime

Month_dict = {
    'Jan': '01',
    'Feb': '02',
    'Mar': '03',
    'Apr': '04',
    'May': '05',
    'Jun': '06',
    'Jul': '07',
    'Aug': '08',
    'Sep': '09',
    'Oct': '10',
    'Nov': '11',
    'Dec': '12'
}

def UTC2timestamp(timestamp):
    _, month, day, hour_min_sec, _, year = timestamp.split(' ')  
    hour, minute, second = hour_min_sec.split(':')   
    return  year + '-' + Month_dict[month] + '-' + day + 'T' + hour + ':' + minute + ':' + second + 'Z'

# return (user item timestamp) sort in get_interaction
def Check_ins(dataset_name):
    '''
    UserID - ID of the user, e.g. 470
    VenueID - ID of the poi, e.g. 49bbd6c0f964a520f4531fe3
    Venue category ID - ID of the poi, e.g. 4bf58dd8d48988d127951735
    Venue category name - name of the poi, e.g. Bridge
    Latitude, e.g. 40.60679958140643
    Longitude e.g. -73.88307005845945
    Timezone offset in minutes e.g. -240
    UTC time  e.g. Tue Apr 03 18:00:09 +0000 2012
    '''
    datas = []
    data_flie = 'data/dataset_TSMC2014_' + dataset_name + '.txt'
    
    f = open(data_flie, 'r', encoding='ISO-8859-1')
    lines = f.readlines()
    for i, line in enumerate(lines):
        user, item, _, _, lat, lon, _, timestamp = line.rstrip().split('\t')
        time = (datetime.strptime(UTC2timestamp(timestamp), "%Y-%m-%dT%H:%M:%SZ") - datetime(1970, 1, 1)).total_seconds() // 60
        coord = ','.join([lat, lon])
        datas.append((user, item, coord, int(time)))

    return datas

def get_interaction(datas):
    user_seq = {}
    for data in datas:
        user, item, coord, time = data
        if user in user_seq:
            user_seq[user].append((item, coord, time))
        else:
            user_seq[user] = []
            user_seq[user].append((item, coord, time))

    for user, item_coord_time in user_seq.items():
        # item_coord_time.sort(key=lambda x: x[-1])  # 对各个数据集得单独排序
        user_seq[user] = sorted(item_coord_time, key=lambda x: x[-1])
    return user_seq

# K-core user_core item_core
def check_Kcore(user_items, user_core, item_core):
    user_count = defaultdict(int)
    item_count = defaultdict(int)
    for user, items in user_items.items():  # items: [[item_id, coord, time], ...]
        for item in items:
            user_count[user] += 1
            item_count[item[0]] += 1

    for user, num in user_count.items():
        if num < user_core:
            return user_count, item_count, False
    for item, num in item_count.items():
        if num < item_core:
            return user_count, item_count, False
    
    return user_count, item_count, True # 已经保证Kcore

# 循环过滤 K-core
def filter_Kcore(user_items, user_core, item_core): # user 接所有items
    user_count, item_count, isKcore = check_Kcore(user_items, user_core, item_core)
    while not isKcore:
        for user, num in user_count.items():
            if num < user_core: # 直接把user 删除
                user_items.pop(user)
            else:
                for item in user_items[user]:
                    if item_count[item[0]] < item_core:
                        user_items[user].remove(item)
                        
        user_count, item_count, isKcore = check_Kcore(user_items, user_core, item_core)
    
    return user_items


def id_map(user_items): # user_items dict

    user2id = {} # raw 2 uid
    item2id = {} # raw 2 iid
    id2user = {} # uid 2 raw
    id2item = {} # iid 2 raw
    user_id = 1  # start from 1
    item_id = 1
    final_data = {}
    
    for user, items in user_items.items():
        if user not in user2id:
            user2id[user] = str(user_id)
            id2user[str(user_id)] = user
            user_id += 1
        iids = [] # item id lists
        for item in items:
            if item[0] not in item2id:
                item2id[item[0]] = str(item_id)
                id2item[str(item_id)] = item[0]
                item_id += 1
            iids.append([item2id[item[0]], item[1], item[2]])
        uid = user2id[user]
        final_data[uid] = iids
    
    data_maps = {
        'user2id': user2id,
        'item2id': item2id,
        'id2user': id2user,
        'id2item': id2item
    }
    
    return final_data, user_id-1, item_id-1, data_maps


def main(data_name):
    np.random.seed(12345)
    # user 5-core item 5-core
    user_core = 5
    item_core = 5

    datas = Check_ins(data_name)

    user_items = get_interaction(datas)
    print(f'{data_name} Raw data has been processed!')
    
    user_items = filter_Kcore(user_items, user_core=user_core, item_core=item_core)
    print(f'User {user_core}-core complete! Item {item_core}-core complete!')

    user_items, user_num, item_num, data_maps = id_map(user_items)  # new_num_id
    user_count, item_count, _ = check_Kcore(user_items, user_core=user_core, item_core=item_core)
    
    user_count_list = list(user_count.values())
    user_avg, user_min, user_max = np.mean(user_count_list), np.min(user_count_list), np.max(user_count_list)
    
    item_count_list = list(item_count.values())
    item_avg, item_min, item_max = np.mean(item_count_list), np.min(item_count_list), np.max(item_count_list)
    
    interact_num = np.sum([x for x in user_count_list])
    sparsity = (1 - interact_num / (user_num * item_num)) * 100
    show_info = f'Total User: {user_num}, Avg User: {user_avg:.4f}, Min Len: {user_min}, Max Len: {user_max}\n' + \
                f'Total Item: {item_num}, Avg Item: {item_avg:.4f}, Min Inter: {item_min}, Max Inter: {item_max}\n' + \
                f'Iteraction Num: {interact_num}, Sparsity: {sparsity:.2f}%'
    print(show_info)

    # -------------- Save Data ---------------
    data_file = 'data/four-'+ data_name + '.txt'

    with open(data_file, 'w') as out:
        for user, items in user_items.items():
            for item in items:
                item_id, item_coord, item_time = item
                out.write(user + '\t' + item_id + '\t' + item_coord + '\t' + str(item_time) + '\t' + '\n')

check_ins_data = ['NYC', 'TKY']

for name in check_ins_data:
    main(name)