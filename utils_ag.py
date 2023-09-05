import sys
import copy
import torch
import random
import pickle
import numpy as np
from math import radians, cos, sin, asin, sqrt
from tqdm import tqdm
from collections import defaultdict
import scipy.sparse as sp
from dataset import Testdataset, Validdataset
from torch.utils.data import DataLoader

def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r

def compute_temPos(time_seq, time_span):
    
    size = time_seq.shape[0]  # maxlen
    time_matrix = np.zeros([size, size], dtype=np.int32)
    for i in range(size):
        for j in range(size):
            span = abs(time_seq[i]-time_seq[j])
            if span > time_span:  # time_span = 256  time threshold in Equation (12)
                time_matrix[i][j] = time_span
            else:
                time_matrix[i][j] = span
    return time_matrix

def Relation_tem(user_train, usernum, maxlen, time_span):
    data_train = dict()
    for user in tqdm(range(1, usernum+1), desc='Preparing relation matrix'):  # user_id starts from 1
        time_seq = np.zeros([maxlen], dtype=np.int32)  # (50, ) "zero" means padding
        idx = maxlen - 1
        for i in reversed(user_train[user][:-1]):  #* select recent maxlen + 1 check-ins as train data
            time_seq[idx] = i[1]
            idx -= 1
            if idx == -1: break
        data_train[user] = compute_temPos(time_seq, time_span)  #* compute the time distance between two different items in train sequence
    return data_train

def compute_disPos(dis_seq, dis_span):
    dis_span = dis_span
    size = len(dis_seq)
    dis_matrix = np.zeros([size, size], dtype=np.float64)
    for i in range(size):
        for j in range(size):
            # print(time_seq)
            lon1 = float(dis_seq[i].split(',')[0])
            lat1 = float(dis_seq[i].split(',')[1])
            lon2 = float(dis_seq[j].split(',')[0])
            lat2 = float(dis_seq[j].split(',')[1])
            span = int(abs(haversine(lon1, lat1, lon2, lat2)))  # equation (13)
            if dis_seq[i] == '0,0' or dis_seq[j] == '0,0':  # padding
                dis_matrix[i][j] = dis_span
            elif span > dis_span:
                dis_matrix[i][j] = dis_span
            else:
                dis_matrix[i][j] = span
    return dis_matrix

def Relation_dis(user_train, usernum, maxlen, dis_span):
    data_train = dict()
    for user in tqdm(range(1, usernum + 1), desc='Preparing dis relation matrix'):
        dis_seq = ['0,0'] * maxlen
        idx = maxlen - 1
        for i in reversed(user_train[user][:-1]):
            dis_seq[idx] = i[2]
            idx -= 1
            if idx == -1: break
        data_train[user] = compute_disPos(dis_seq, dis_span)
    return data_train

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
    for user, items in User.items():  # each user's check-ins
        user_set.add(user)  # [[loc_id, time, gps], ...]
        User_filted[user] = items
        for item in items:
            item_set.add(item[0])
    user_map = dict()
    item_map = dict()
    for u, user in enumerate(user_set):  # start from 1, i.e., 1~4638
        user_map[user] = u+1
    for i, item in enumerate(item_set):  # same as above, i.e., 1~9731
        item_map[item] = i+1
    
    for user, items in User_filted.items():  # sort check-ins chronologically
        User_filted[user] = sorted(items, key=lambda x: x[1])  

    User_res = dict()
    for user, items in User_filted.items():
        User_res[user_map[user]] = list(map(lambda x: [item_map[x[0]], time_map[x[1]], x[2]], items))

    time_max = set()
    for user, items in User_res.items():
        time_list = list(map(lambda x: x[1], items))  # the list of each check-in's time slot of each user
        time_diff = set()
        for i in range(len(time_list)-1):
            if time_list[i+1] - time_list[i] != 0:  #* record the difference between consecutive items
                time_diff.add(time_list[i+1] - time_list[i])
        if len(time_diff)==0:
            time_scale = 1
        else:
            time_scale = min(time_diff)  # minimal consecutive time difference of each user as scale
        time_min = min(time_list)  # each user's minimal time slot
        User_res[user] = list(map(lambda x: [x[0], int(round((x[1]-time_min)/time_scale)+1), x[2]], items))  # Equation (12)
        time_max.add(max(set(map(lambda x: x[1], User_res[user]))))  # each user's maximal scaled time slot

    return User_res, len(user_set), len(item_set), max(time_max)

def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    
    print('Preparing data...')
    f = open('data/%s.txt' % fname, 'r')
    time_set = set()

    user_count = defaultdict(int)
    item_count = defaultdict(int)
    for line in f:  #* count user check-ins count and item count statistics for filtering unpopular users and items
        try:
            u, i, location, timestamp = line.rstrip().split('\t')
        except:
            u, i, timestamp = line.rstrip().split('\t')
        u = int(u)
        i = int(i)
        user_count[u]+=1  # 1~4368
        item_count[i]+=1  # 1~9731
    f.close()
    f = open('data/%s.txt' % fname, 'r')

    for line in f:
        try:
            u, i, location, timestamp = line.rstrip().split('\t')
        except:
            u, i, timestamp = line.rstrip().split('\t')
        u = int(u)
        i = int(i)
        timestamp = float(timestamp)
        if user_count[u]<5 or item_count[i]<5:
            continue
        time_set.add(timestamp)  # record time that all check-ins happened
        User[u].append([i, timestamp, location])  # record user check-ins
    f.close()
    time_map = timeSlice(time_set)  # record each time interval between it and time_min
    User, usernum, itemnum, timenum = cleanAndsort(User, time_map)  # users' check-ins, usernum, itemnum, and maximal scaled time slot among all users' check-ins 

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:  #* useless condition?! yes, because inactive users with less 5 check-ins are filtered out
            user_train[user] = User[user]
            # user_valid[user] = []
            # user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])  # one check-in record
            user_test[user] = []
            user_test[user].append(User[user][-1])  # same
    print('Preparing done...')
    return [user_train, user_valid, user_test, usernum, itemnum, timenum]

def sparse_dropout(x, rate, noise_shape):

    random_tensor = 1-rate
    random_tensor += torch.rand(noise_shape).to(x.device)
    dropout_mask = torch.floor(random_tensor).bool()
    i = x._indices()
    v = x._values()

    i = i[:,dropout_mask]
    v = v[dropout_mask]

    out = torch.sparse.FloatTensor(i,v,x.shape).to(x.device)

    out = out*(1./(1-rate))

    return out

def get_adj_matrix(matrix):

    row_sum = np.array(matrix.sum(1)) + 1e-24
    degree_mat_inv_sqrt = sp.diags(np.power(row_sum, -0.5).flatten())
    rel_matrix_normalized = degree_mat_inv_sqrt.dot(matrix.dot(degree_mat_inv_sqrt)).todense()
    return rel_matrix_normalized


def generate_vaild(dataset, args):
    [train, valid, test, usernum, itemnum, timenum] = copy.deepcopy(dataset)
    users = range(1, usernum + 1)
    all_vaild_user = []
    all_vaild_seq = []
    # all_test_time_seq = []
    all_vaild_time_matrix = []
    all_vaild_dis_matrix = []
    all_labels = []

    for u in users:
        if u % 1000 == 0:
            print('.', end='')
            sys.stdout.flush()
        if len(train[u]) < 1 or len(valid[u]) < 1: continue
        seq = np.zeros([args.maxlen], dtype=np.int32)
        time_seq = np.zeros([args.maxlen], dtype=np.int32)
        dis_seq = ['0,0'] * args.maxlen
        idx = args.maxlen - 1

        for i in reversed(train[u]):
            seq[idx] = i[0]
            time_seq[idx] = i[1]
            dis_seq[idx] = i[2]
            idx -= 1
            if idx == -1: break
        time_matrix = compute_temPos(time_seq, args.time_span)
        dis_matrix = compute_disPos(dis_seq, args.dis_span)
        all_vaild_user.append(u)
        all_vaild_seq.append(seq)
        all_vaild_time_matrix.append(time_matrix)
        all_vaild_dis_matrix.append(dis_matrix)
        all_labels.append(valid[u][0][0])

    with open(f"data/{args.dataset}_" + '_vaild_instance.pkl','wb') as f:
        pickle.dump(all_vaild_user, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(all_vaild_seq, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(all_vaild_time_matrix, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(all_vaild_dis_matrix, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(all_labels, f, pickle.HIGHEST_PROTOCOL)

def evaluate_vaild(model, dataset, args):
    NDCG = [0.0, 0.0, 0.0]
    vaild_user_num = 0.0
    HT = [0.0, 0.0, 0.0]
    try:
        with open(f"data/{args.dataset}_" + "vaild_instance.pkl", 'rb') as f:
            all_u = pickle.load(f)
            all_seqs = pickle.load(f)
            all_time_matrix = pickle.load(f)
            all_distance_matrix = pickle.load(f)
            all_labels = pickle.load(f)
    except:
        print('Preparing vaild instances')
        generate_vaild(dataset, args)
        with open(f"data/{args.dataset}_" + "vaild_instance.pkl", 'rb') as f:
            all_u = pickle.load(f)
            all_seqs = pickle.load(f)
            all_time_matrix = pickle.load(f)
            all_distance_matrix = pickle.load(f)
            all_labels = pickle.load(f)

    vaild_dataset = Validdataset(all_seqs, all_time_matrix, all_distance_matrix,all_labels)
    dataloader = DataLoader(dataset=vaild_dataset,batch_size=args.batch_size,shuffle=False)

    for instance in dataloader:
        print('.', end='')
        sys.stdout.flush()
        u,seq,time_matrix,dis_matrix,label = instance
        predictions,no_use = model.predict(u,seq,time_matrix,dis_matrix,[1])
        predictions = -predictions
        ranks = predictions.argsort().argsort().cpu()
        rank = []
        for i in range(len(ranks)):
            rank.append(ranks[i,label[i]])
        vaild_user_num += len(rank)
        for i in rank:
            if i < 2:
                NDCG[0] += 1 / np.log2(i + 2)
                HT[0] += 1
            if i < 5:
                NDCG[1] += 1 / np.log2(i + 2)
                HT[1] += 1
            if i < 10:
                NDCG[2] += 1 / np.log2(i + 2)
                HT[2] += 1

    return [x/vaild_user_num for x in NDCG], [x/vaild_user_num for x in HT]

def generate_test(dataset,args):
    [train, valid, test, usernum, itemnum, timenum] = copy.deepcopy(dataset)
    users = range(1, usernum + 1)
    all_test_user = []
    all_test_seq = []
    # all_test_time_seq = []
    all_test_time_matrix = []
    all_test_dis_matrix = []
    all_labels = []

    for u in users:
        if u % 1000 ==0:
            print('.', end='')
            sys.stdout.flush()
        if len(train[u]) < 1 or len(test[u]) < 1: continue
        seq = np.zeros([args.maxlen], dtype=np.int32)
        time_seq = np.zeros([args.maxlen], dtype=np.int32)
        dis_seq = ['0,0'] * args.maxlen
        idx = args.maxlen - 1
        seq[idx] = valid[u][0][0]
        time_seq[idx] = valid[u][0][1]
        dis_seq[idx] = valid[u][0][2]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i[0]
            time_seq[idx] = i[1]
            dis_seq[idx] = i[2]
            idx -= 1
            if idx == -1: break
        time_matrix = compute_temPos(time_seq, args.time_span)
        dis_matrix = compute_disPos(dis_seq,args.dis_span)
        all_test_user.append(u)
        all_test_seq.append(seq)
        # all_test_time_seq.append(time_seq)
        all_test_time_matrix.append(time_matrix)
        all_test_dis_matrix.append(dis_matrix)
        all_labels.append(test[u][0][0])

    with open(f"data/{args.dataset}_" + 'test_instance.pkl','wb') as f:
        pickle.dump(all_test_user, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(all_test_seq, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(all_test_time_matrix, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(all_test_dis_matrix, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(all_labels, f, pickle.HIGHEST_PROTOCOL)

def evaluate_test(model, dataset, args):
    NDCG = [0.0, 0.0, 0.0]
    test_user_num = 0.0
    HT = [0.0, 0.0, 0.0]
    try:
        with open(f"data/{args.dataset}_" + "test_instance.pkl", 'rb') as f:
            all_u = pickle.load(f)
            all_seqs = pickle.load(f)
            all_time_matrix = pickle.load(f)
            all_distance_matrix = pickle.load(f)
            all_labels = pickle.load(f)
    except:
        print('Preparing test instances')
        generate_test(dataset, args)
        with open(f"data/{args.dataset}_" + "test_instance.pkl", 'rb') as f:
            all_u = pickle.load(f)
            all_seqs = pickle.load(f)
            all_time_matrix = pickle.load(f)
            all_distance_matrix = pickle.load(f)
            all_labels = pickle.load(f)
    
    test_dataset = Testdataset(all_seqs, all_time_matrix, all_distance_matrix, all_labels)
    dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    for instance in dataloader:
        print('.', end='')
        sys.stdout.flush()
        u, seq, time_matrix, dis_matrix, label = instance
        predictions, no_use = model.predict(u, seq, time_matrix, dis_matrix, [1]) 
        predictions = -predictions  
        ranks = predictions.argsort().argsort().cpu()
        rank = []
        for i in range(len(ranks)):
            rank.append(ranks[i, label[i]])
        test_user_num += len(rank)
        for i in rank:
            if i < 2:
                NDCG[0] += 1 / np.log2(i + 2)
                HT[0] += 1
            if i < 5:
                NDCG[1] += 1 / np.log2(i + 2)
                HT[1] += 1
            if i < 10:
                NDCG[2] += 1 / np.log2(i + 2)
                HT[2] += 1

    return [x/test_user_num for x in NDCG], [x/test_user_num for x in HT]