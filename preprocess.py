import argparse
from collections import defaultdict
from datetime import datetime
import intervals as I

def filter_latlon(lat, lon, limit):
    # return lat not in I.closed(limit[0][0], limit[0][1]) or lat not in I.closed(limit[2][0], limit[2][1]) \
    # or lon not in I.closed(limit[1][0], limit[1][1]) or lon not in I.closed(limit[3][0], limit[3][1])
    return lat not in I.closed(limit[0][0], limit[0][1]) or lon not in I.closed(limit[1][0], limit[1][1]) 

def timeSlice(time_set):
    time_min = min(time_set)  
    time_map = dict()
    for time in time_set:
        time_map[time] = int(round(float(time-time_min)))
    return time_map  # record each time interval between it and time_min

def cleanAndsort(User, time_map):
    User_filted = dict()
    user_set = set()
    item_set = set()
    for user, items in User.items():  # each user's check-ins
        user_set.add(user)  
        User_filted[user] = items  # [[loc_id, gps, time], ...]
        for item in items:
            item_set.add(item[0])
    user_map = dict()
    item_map = dict()
    for u, user in enumerate(user_set):  # start from 1, i.e., 1~4638
        user_map[user] = u+1
    for i, item in enumerate(item_set):  # same as above, i.e., 1~9731
        item_map[item] = i+1
    
    for user, items in User_filted.items():  # sort check-ins chronologically
        User_filted[user] = sorted(items, key=lambda x: x[-1])  

    User_res = dict()
    for user, items in User_filted.items():
        User_res[user_map[user]] = list(map(lambda x: [item_map[x[0]], time_map[float(x[2])], x[1]], items))  # [[loc_id, time, gps], ...]

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

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='gowalla',type=str)  
parser.add_argument('--user_threshold', default=5, type=int) 
parser.add_argument('--item_threshold', default=5, type=int) 
args = parser.parse_args()

if __name__ == '__main__':
    print('Preparing data...')
    f = open('data/loc-%s_edges.txt' % args.dataset, 'r')  # f = open('data/checkins-%s.txt' % args.dataset, 'r')  
    f_w = open('data/%s_CalNev.txt' % args.dataset, 'w')  # extract dataset from March 2009 to October 2010 in California and Nevada
    time_limit = datetime(2009, 3, 1)
    # [(32°32′ N to 42° N), (114°8′ W to 124°26′ W), (35° N to 42° N), (114° 2′ W to 120° W)]  #* result from wikipedia
    # distance_limit = [(32.53333333333333, 42.7), (-123.56666666666666, -113.86666666666666), (35.583333333333336, 42.7), (-122, -115.9)]  # latitude and longitude range of California and Nevada
    # distance_limit = [(32.32, 42), (-124.26, -114.8), (35, 42), (-120, -114.2)]
    distance_limit = [(32.32, 42), (-124.26, -114.2)]  # (-123.56666666666666, -113.86666666666666):{11742, 35833},(-123.56666666666666, -115.9):{10694, 33187},(-122, -115.9):{7374, 20955},(-122, -113.86666666666666):{8601, 23601}
    print(distance_limit)
    # filter dataset
    for line in f:  
        u, timestamp, lat, lon, i = line.rstrip().split('\t')  
        lat_, lon_ = float(lat), float(lon)  # latitude and longitude
        timestamp = (datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ") - time_limit).total_seconds()  # unix seconds
        if timestamp < 0 or filter_latlon(lat_, lon_, distance_limit):
            continue
        f_w.write('\t'.join([u, str(timestamp), lat, lon, i]) + ' \n')
    f.close()
    f_w.close()
    
    user_count = defaultdict(int)
    item_count = defaultdict(int)
    f = open('data/%s_CalNev.txt' % args.dataset, 'r')
    # count user check-ins count and item count statistics for filtering unpopular users and items
    for line in f:  
        u, timestamp, lat, lon, i = line.rstrip().split('\t')
        u = int(u)
        i = int(i)
        user_count[u]+=1  
        item_count[i]+=1  
    print(len(user_count), len(item_count))
    f.close()  
    
    usernum = 0  # gowalla: 107092  85022(2019-7-1)  84297(2019-12-1)  68511(2010-5-1)  51273(2010-8-1)  35063(2010-9-15)  32184(2010-9-20)  34566(2010-9-25)  24953(2010-10-1)  7402(2010-10-15)
    itemnum = 0  # gowalla: 1280969  308877(2019-7-1) 303804(2019-12-1) 208823(2010-5-1)  114099(2010-8-1)  50129(2010-9-15)  42051(2010-9-20)  29301(2010-9-25)  25533(2010-10-1)  4364(2010-10-15)
    user_map = dict()
    item_map = dict()
    time_set = set()
    User = dict()
    f = open('data/%s_CalNev.txt' % args.dataset, 'r')
    for line in f:
        u, timestamp, lat, lon, i = line.rstrip().split('\t')
        u = int(u)
        i = int(i)
        if user_count[u]<args.user_threshold or item_count[i]<args.item_threshold:
            continue
        if u in user_map:
            userid = user_map[u]
        else:
            usernum += 1
            userid = usernum  # start from 1
            user_map[u] = userid
            User[userid] = []
        if i in item_map:
            itemid = item_map[i]
        else:
            itemnum += 1
            itemid = itemnum  # start from 1
            item_map[i] = itemid
        coord = ','.join([lat, lon])
        User[userid].append([itemid, coord, timestamp])
        time_set.add(float(timestamp))  # record time that all check-ins happened
    f.close()

    print('time max: ', max(time_set))
    print('time min: ', min(time_set))
    print('user count: ', len(user_map))
    print('item count: ', len(item_map))
    # sort pois in User according to time
    for userid in User.keys():
        User[userid].sort(key=lambda x: x[-1])
      
    print('Preparing done...')
    time_map = timeSlice(time_set)  # record each time interval between it and time_min
    User, usernum, itemnum, timenum = cleanAndsort(User, time_map)  # users' check-ins, usernum, itemnum, and maximal scaled time slot among all users' check-ins 
    cc = 0.
    for u in User:
        cc += len(User[u])
    print(usernum)
    print(itemnum)
    print('average sequence length: %.2f' % (cc / len(User)))
    
    # f = open('data/%s.txt' % args.dataset, 'w')
    # for user in User.keys():
    #     for i in User[user]:
    #         f.write('%s\t%s\t%s\t%s\n' % (user, i[0], i[1],i[2],))
    # f.close()   