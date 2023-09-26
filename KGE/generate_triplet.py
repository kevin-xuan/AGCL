#  先构造user-item，划分train/test集
#  定义relation2id.txt
#  根据user/item构造entity2id.txt
#  然后构造train/test 三元组
#  去掉重复三元组

import os
from math import radians, cos, sin, asin, sqrt
from tqdm import tqdm
from collections import defaultdict
from constant import DATA_NAME, SCHEME
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='gowalla',type=str)  # 'four-sin'  'gowalla'
args = parser.parse_args()

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    lat1, lon1, lat2, lon2 = float(lat1), float(lon1), float(lat2), float(lon2)
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r


def generate_train_test_checkin(train_file):
    if os.path.exists(train_file):
        print('train.txt has existed!!!')
        return

    with open(train_file, 'w+') as f_train:
        for u, items in user_train.items():
            train_locs = list(map(lambda x: x[0]-1, items))  # start from 0
            train_locs.insert(0, u-1)

            for train_elem in train_locs:
                f_train.write(str(train_elem) + ' ')
            f_train.write('\n')
            
    print('Successfully generate train checkins!')


def generate_entity_file(entity2id_file):  # 构造entity2id文件
    if os.path.exists(entity2id_file):
        print('entity2id.txt has existed!!!')
        return
    
    with open(entity2id_file, 'w+') as f:
        for i in range(user_num):  # start from 0
            f.write(str(i) + ' ')
            f.write(str(i) + ' ')
            f.write('\n')
        for l in range(item_num):
            poi_id = l + user_num
            f.write(str(poi_id) + ' ')
            f.write(str(poi_id) + ' ')
            f.write('\n')
    print('Successfully generate entity2id.txt!')


def generate_train_triplets(train_file, train_triplets_file):  # 构造train 三元组
    f_train_triplets = open(train_triplets_file, 'w+')
    # print('Construct interact relation and temporal relation......')
    with tqdm(total=user_num) as bar:
        with open(train_file, 'r') as f:
            for line in f.readlines():
                line = line.strip().split(' ')  # 以空格形式分隔
                user_id = line[0]  # str
                poi_ids = line[1:]

                # 构建interact关系
                for poi_id in poi_ids:
                    poi_id = str(int(poi_id) + user_num)
                    f_train_triplets.write(user_id + '\t')
                    f_train_triplets.write(poi_id + '\t')
                    f_train_triplets.write('0' + '\n')  # 0代表interact relation

                # 构建temporal关系  相邻poi相连
                for i in range(len(poi_ids) - 1):
                    poi_prev = str(int(poi_ids[i]) + user_num)
                    poi_next = str(int(poi_ids[i + 1]) + user_num)
                    if poi_prev != poi_next:
                        f_train_triplets.write(poi_prev + '\t')
                        f_train_triplets.write(poi_next + '\t')
                        f_train_triplets.write('1' + '\n')  # 1代表temporal relation
                bar.update(1)

    # 构建spatial关系  两个poi的距离小于距离阈值lambda_d，就相连
    pois_list = []
    for poi, coord in gps_map.items(): 
        pois_list.append((poi-1, coord))  # start from 0

    # 方案1
    if SCHEME == 1:
        lambda_d = 0.2  # 距离阈值为0.2千米
        with tqdm(total=len(pois_list)) as bar:
            for i in range(len(pois_list)):
                for j in range(i+1, len(pois_list)):
                    poi_prev, coord_prev = pois_list[i]
                    poi_next, coord_next = pois_list[j]
        
                    poi_prev = poi_prev + user_num  # poi在entity中所对应的实体集
                    poi_next = poi_next + user_num
                    lat_prev, lon_prev = coord_prev.split(',')  # "0.,0."
                    lat_next, lon_next = coord_next.split(',')
        
                    dist = haversine(lat_prev, lon_prev, lat_next, lon_next)
                    if dist <= lambda_d:
                        f_train_triplets.write(str(poi_prev) + '\t')
                        f_train_triplets.write(str(poi_next) + '\t')
                        f_train_triplets.write('2' + '\n')  # 2代表spatial relation
                        # spatial relation是对称的
                        f_train_triplets.write(str(poi_next) + '\t')
                        f_train_triplets.write(str(poi_prev) + '\t')
                        f_train_triplets.write('2' + '\n')
        
                bar.update(1)

    # 方案2
    else:
        lambda_d = 3  # 距离阈值为3千米, 再取top k, 即双重限制
        with tqdm(total=len(pois_list)) as bar:
            for i in range(len(pois_list)):
                loci_list = []
                for j in range(len(pois_list)):
                    poi_prev, coord_prev = pois_list[i]
                    poi_next, coord_next = pois_list[j]

                    poi_prev = poi_prev + user_num   
                    poi_next = poi_next + user_num 
                    lat_prev, lon_prev = coord_prev.split(',')
                    lat_next, lon_next = coord_next.split(',')

                    dist = haversine(lat_prev, lon_prev, lat_next, lon_next)
                    if dist <= lambda_d and poi_prev != poi_next:
                        loci_list.append((poi_next, dist))  # 先是第一重限制, 这样可能会造成很多重复计算

                sort_list = sorted(loci_list, key=lambda x: x[1])  # 从小到大排序,距离越小,排名越靠前
                length = min(len(sort_list), 50)
                select_pois = sort_list[:length]  # 一般情况下, sort_list的长度肯定不止50, 取top 50  这是第二重限制
                for poi_entity, _ in select_pois:
                    f_train_triplets.write(str(poi_prev) + '\t')
                    f_train_triplets.write(str(poi_entity) + '\t')
                    f_train_triplets.write('2' + '\n')  # 2代表spatial relation
                    # spatial relation是对称的
                    f_train_triplets.write(str(poi_entity) + '\t')
                    f_train_triplets.write(str(poi_prev) + '\t')
                    f_train_triplets.write('2' + '\n')

                bar.update(1)
                
    f_train_triplets.close()


# 可能会重复添加triplet，所以要进行去重操作，得到最终train triplets
def filter_train_triplet(read_file, write_file):
    filter_set = set()
    print('Filter repeated triplets......')
    count = 0
    with open(read_file, 'r') as f_read, open(write_file, 'w+') as f_write:
        for f_read_line in f_read.readlines():
            count += 1
            f_read_line = f_read_line.strip('\n')
            if f_read_line not in filter_set:
                filter_set.add(f_read_line)
        for triplet in filter_set:
            f_write.write(triplet + '\n')
    print('Original triplets: ', count)
    print('Final triplets: ', len(filter_set))
    return filter_set


# 去重且保证test triplets与train triplets不同
def filter_test_triplet(read_file, write_file, train_filter_set):
    filter_set = set()
    print('Filter repeated triplets......')
    count = 0
    with open(read_file, 'r') as f_read, open(write_file, 'w+') as f_write:
        for f_read_line in f_read.readlines():
            count += 1
            f_read_line = f_read_line.strip('\n')
            if f_read_line not in filter_set and f_read_line not in train_filter_set:
                filter_set.add(f_read_line)
        for triplet in filter_set:
            f_write.write(triplet + '\n')
    print('Original triplets: ', count)
    print('Final triplets: ', len(filter_set))

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
    gps_map = dict()
    for u, user in enumerate(user_set):
        user_map[user] = u+1
    for i, item in enumerate(item_set): 
        item_map[item] = i+1
    
    for user, items in User_filted.items():  # sort check-ins chronologically
        User_filted[user] = sorted(items, key=lambda x: x[1])  

    User_res = dict()
    for user, items in User_filted.items():
        User_res[user_map[user]] = list(map(lambda x: [item_map[x[0]], time_map[x[1]], x[2]], items))
        for x in items:
            poi_id = item_map[x[0]]
            if gps_map.get(poi_id, None) is None:
                gps_map[poi_id] = x[2]
        
    user_num = len(user_set)
    item_num = len(item_set)

    return User_res, user_num, item_num, gps_map

def data_partition(fname):
    User = defaultdict(list)
    user_train = {}
    
    print('Preparing data...')
    user_count = defaultdict(int)
    item_count = defaultdict(int)
    time_set = set()
    f = open('../data/%s.txt' % fname, 'r')
    for line in f:  #* count user check-ins count and item count statistics for filtering unpopular users and items
        try:
            u, i, location, timestamp = line.rstrip().split('\t')
        except:
            u, i, timestamp = line.rstrip().split('\t')
        u = int(u)
        i = int(i)
        user_count[u]+=1  # foursquare: 1~4368
        item_count[i]+=1  # foursquare: 1~9731
    f.close()
    
    f = open('../data/%s.txt' % fname, 'r')
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
    User, user_num, item_num, gps_map = cleanAndsort(User, time_map) 

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:  #* useless condition?! yes, because inactive users with less 5 check-ins are filtered out
            user_train[user] = User[user]
        else:
            user_train[user] = User[user][:-2]
            
    return user_train, user_num, item_num, gps_map

if __name__ == '__main__':
    user_train, user_num, item_num, gps_map = data_partition(args.dataset)
    
    print('Active POI number: ', item_num)  
    print('Active User number: ', user_num)  

    data_path = './dataset/{}/{}_scheme{}'.format(DATA_NAME, DATA_NAME, SCHEME)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    # generate train & test file
    train_file = os.path.join(data_path, 'train.txt')
    entity2id_file = os.path.join(data_path, 'entity2id.txt')
    train_triplets = os.path.join(data_path, 'train_triplets.txt')
    final_train_triplets = os.path.join(data_path, 'final_train_triplets.txt')

    print('Generate train/test checkins......')
    generate_train_test_checkin(train_file)
    print('Generate entity2id......')
    generate_entity_file(entity2id_file)
    print('Construct train triplets......')
    generate_train_triplets(train_file, train_triplets)  # 生成train

    train_filter_triplets = filter_train_triplet(train_triplets, final_train_triplets)  # train三元组去重