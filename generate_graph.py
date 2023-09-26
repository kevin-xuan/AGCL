import argparse
from collections import defaultdict
from tqdm import tqdm 
from scipy.sparse import csr_matrix, dok_matrix
import numpy as np
import scipy.sparse as sp
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='gowalla',type=str)  
args = parser.parse_args()

def normalize(mx):
    '''Row-normalize sparse matrix'''
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum.astype(float), -1).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def timeSlice(time_set):
    time_min = min(time_set)  # 1090.0 
    time_map = dict()
    for time in time_set:
        time_map[time] = int(round(float(time-time_min)))
    return time_map  # record each time interval between it and time_min

def cleanAndsort(User, time_map):
    User_filted = dict()
    user_set = set()
    item_set = set()
    for user, items in User.items():
        user_set.add(user)  # [[loc_id, time, gps], ...]
        User_filted[user] = items
        for item in items:
            item_set.add(item[0])
    user_map = dict()
    item_map = dict()
    for u, user in enumerate(user_set):  # start from 0
        user_map[user] = u+1
    for i, item in enumerate(item_set): 
        item_map[item] = i+1
    
    for user, items in User_filted.items():  # sort check-ins chronologically
        User_filted[user] = sorted(items, key=lambda x: x[1])  

    User_res = dict()
    for user, items in User_filted.items():
        User_res[user_map[user]] = list(map(lambda x: [item_map[x[0]], time_map[x[1]], x[2]], items))
    user_num = len(user_set)
    item_num = len(item_set)

    return User_res, user_num, item_num

def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    
    print('Preparing data...')
    user_count = defaultdict(int)
    item_count = defaultdict(int)
    user2id  = defaultdict(int)
    item2id  = defaultdict(int)
    time_set = set()
    f = open('data/checkins-%s.txt' % fname, 'r')  #* Graph-Flashback
    for line in f:  #* count user check-ins count and item count statistics for filtering unpopular users and items
        try:
            # u, i, location, timestamp = line.rstrip().split('\t')
            u, timestamp, lat, long, i = line.rstrip().split('\t')
        except:
            u, i, timestamp = line.rstrip().split('\t')
        u = int(u)
        i = int(i)
        #* graph-flashback mapping
        if user2id.get(u) is None:
            user2id[u] = len(user2id)
        if item2id.get(i) is None:
            item2id[i] = len(item2id)
        # user_count[u]+=1  # foursquare: 1~4368  gowalla: 1~10915
        # item_count[i]+=1  # foursquare: 1~9731  gowalla: 1~33805
        user_count[user2id[u]]+=1  # foursquare: 1~4368  gowalla: 1~10915
        item_count[item2id[i]]+=1  # foursquare: 1~9731  gowalla: 1~33805
    f.close()
    
    f = open('data/checkins-%s.txt' % fname, 'r')  #* Graph-Flashback
    for line in f:
        try:
            # u, i, location, timestamp = line.rstrip().split('\t')
            u, timestamp, lat, long, i = line.rstrip().split('\t')
            timestamp =(datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ") - datetime(1970, 1,
                                                                                  1)).total_seconds()  # unix seconds
            location = ','.join([lat, long])
        except:
            u, i, timestamp = line.rstrip().split('\t')
        u = int(u)
        i = int(i)
        timestamp = float(timestamp)
        # if user_count[u]<5 or item_count[i]<5:
        #     continue
        if user_count[user2id[u]]<5 or item_count[item2id[i]]<5:
            continue
        time_set.add(timestamp)  # record time that all check-ins happened
        User[user2id[u]].append([item2id[i], timestamp, location])  # record user check-ins
    f.close()
    
    time_map = timeSlice(time_set)  # record each time interval between it and time_min
    User, user_num, item_num = cleanAndsort(User, time_map) 
    f = open('data/checkins-%s_agran.txt' % args.dataset, 'w')
    for user in User.keys():
        for i in User[user]:
            f.write('%s\t%s\t%s\t%s\n' % (str(user), str(i[0]), i[2], i[1]))  # (user, item, coord, timestamp)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:  #* useless condition?! yes, because inactive users with less 5 check-ins are filtered out
            user_train[user] = User[user]
        else:
            user_train[user] = User[user][:-2]
    
    transition_graph = dok_matrix((item_num, item_num), dtype=np.float32)  
    interaction_graph = dok_matrix((user_num, item_num), dtype=np.float32)
    for user, items in user_train.items():
        user_id = user - 1  # graph index start from 0
        item_list = list(map(lambda x: x[0]-1, items)) 
        for i in range(len(item_list)-1):
            interaction_graph[user_id, item_list[i]] += 1
            transition_graph[item_list[i], item_list[i+1]] += 1
        interaction_graph[user_id, item_list[-1]] += 1
    transition_graph = csr_matrix(normalize(transition_graph.A))
    interaction_graph = csr_matrix(normalize(interaction_graph.A))
    sp.save_npz('data/checkins-%s_transaction_kl_notest.npz' % args.dataset, transition_graph)
    sp.save_npz('data/checkins-%s_interaction_kl_notest.npz' % args.dataset, interaction_graph)
    print('Preparing done...')
    

if __name__ == '__main__':
    
    user_train = data_partition(args.dataset)