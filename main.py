import os
import time
import torch
import pickle
import argparse
import scipy.sparse as sp
from dataset import Traindataset
from torch.utils.data import DataLoader
from model import MAGCN
from tqdm import tqdm
from utils import *
import random
import shutil

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

def setup_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

setup_seed(42)

parser = argparse.ArgumentParser()
# dataset
parser.add_argument('--dataset', default='four-sin', choices=['gowalla', 'four-sin'], type=str, help="dataset name")
# training
parser.add_argument('--train_dir', default='default',type=str, help="log save directory")
parser.add_argument('--batch_size', default=64, type=int, help="batch size")
parser.add_argument('--lr', default=0.001, type=float, help="learning rate")
parser.add_argument('--gpu', default=1, type=int, help="device")
parser.add_argument("--same_anchor", type=eval, choices=[True, False], default='False', help="whether to use same anchor for temporal, spatial and frequency prior")
# model
parser.add_argument('--maxlen', default=50, type=int, help="use recent maxlen subsequence to train model")
parser.add_argument('--hidden_units', default=64, type=int, help="embedding dimension")
parser.add_argument('--num_blocks', default=2, type=int, help="number of transformer layers")
parser.add_argument('--num_epochs', default=60, type=int, help="number fo training epoch")  
parser.add_argument('--num_heads', default=1, type=int, help="head number of multi-head attention")
parser.add_argument('--dropout_rate', default=0.2, type=float, help="drop rate")
parser.add_argument('--l2_emb', default=0.001, type=float, help="L2 regularization")
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str, help="model checkpoint")
parser.add_argument('--time_span', default=128, type=int, help="maximal time interval threshold")  
parser.add_argument('--dis_span', default=256, type=int, help="maximal distance interval threshold")  
parser.add_argument('--anchor_num', default=500, type=int, help="number of anchor node")  
parser.add_argument('--layer_num', default=3, type=int, help="number of GCN layers")
parser.add_argument('--time_slot', default=168, type=int, help="segment time by 24 hours * 7 days")
# loss 
parser.add_argument('--tra_kl_reg', default=1.0, type=float, help="kl_reg of frequency transition prior")
parser.add_argument('--time_kl_reg', default=1.0, type=float, help="kl_reg of temporal transition prior;0 means discarding this loss")
parser.add_argument('--dis_kl_reg', default=1.0, type=float, help="kl_reg of spatial prior;0 means discarding this loss")
parser.add_argument('--contra_reg', default=1.0, type=float, help="kl_reg of graph contrastive loss;0 means discarding this loss")
parser.add_argument('--contra_intra', default=1.0, type=float, help="intra-graph contrastive loss")
parser.add_argument('--contra_inter', default=1.0, type=float, help="inter-graph contrastive loss")
# graph
parser.add_argument('--tra_delta', default=0., type=float, help="mask threshold for adaptive frequency transition graph")
parser.add_argument('--tem_delta', default=0.4, type=float, help="mask threshold for adaptive temporal transition graph")
parser.add_argument('--dis_delta', default=0., type=float, help="mask threshold for adaptive spatial proximity graph")
# evaluation
parser.add_argument('--valid_epoch', default=1, type=int, help="every n epoch to test")


args = parser.parse_args()

def mask(adj, epsilon=0, mask_value=-1e16):
    mask = (adj > epsilon).detach().float()
    update_adj = adj * mask + (1 - mask) * mask_value
    return update_adj

def count_params(model):
    param_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            # print(name, param.shape, param.numel())
            param_count += param.numel()
    print(f'In total: {param_count} trainable parameters.')
    return param_count

if __name__ == '__main__':
    tra_adj_matrix = sp.load_npz('data/%s_transaction_kl_notest.npz' % args.dataset) 
    
    # pre-trained temporal and spatial graph from Graph-Flashback
    if args.dataset == 'four-sin':
        with open('KGE/POI_graph/foursquare_scheme2_transe_loc_spatial_20.pkl', 'rb') as f:  
            dis_adj_matrix = pickle.load(f)  
        with open('KGE/POI_graph/foursquare_scheme2_transe_loc_temporal_20.pkl', 'rb') as f:  
            time_adj_matrix = pickle.load(f)  
    elif args.dataset == 'gowalla':
        with open('KGE/POI_graph/gowalla_scheme2_transe_loc_spatial_100.pkl', 'rb') as f:  
            dis_adj_matrix = pickle.load(f)  
        with open('KGE/POI_graph/gowalla_scheme2_transe_loc_temporal_100.pkl', 'rb') as f:  
            time_adj_matrix = pickle.load(f)  
    elif args.dataset == 'four-NYC':
        with open('KGE/POI_graph/nyc_scheme2_transe_loc_spatial_10.pkl', 'rb') as f:  
            dis_adj_matrix = pickle.load(f)  
        with open('KGE/POI_graph/nyc_scheme2_transe_loc_temporal_10.pkl', 'rb') as f:  
            time_adj_matrix = pickle.load(f)  
    elif args.dataset == 'four-TKY':
        with open('KGE/POI_graph/tky_scheme2_transe_loc_spatial_10.pkl', 'rb') as f:  
            dis_adj_matrix = pickle.load(f)  
        with open('KGE/POI_graph/tky_scheme2_transe_loc_temporal_10.pkl', 'rb') as f:  
            time_adj_matrix = pickle.load(f)  
            
    elif args.dataset == 'gowalla':
        with open('KGE/POI_graph/gowalla_scheme2_transe_loc_spatial_100.pkl', 'rb') as f:  
            dis_adj_matrix = pickle.load(f)  
        with open('KGE/POI_graph/gowalla_scheme2_transe_loc_temporal_100.pkl', 'rb') as f:  
            time_adj_matrix = pickle.load(f)  
    tra_adj_matrix = tra_adj_matrix.todok()
    dis_adj_matrix = dis_adj_matrix.todok()
    time_adj_matrix = time_adj_matrix.todok()
    
    args.device = torch.device("cuda:{}".format(args.gpu)) if torch.cuda.is_available() else torch.device("cpu")

    dataset = data_partition(args.dataset)
    [user_train, user_valid, user_test, usernum, itemnum, timenum, user_interval_map, item_interval_map] = dataset  
    num_batch = len(user_train) // args.batch_size
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print(itemnum)
    print(usernum)
    print('total transition number: %d' % int(cc))
    print('average sequence length: %.2f' % (cc / len(user_train)))

    if not os.path.isdir(args.dataset + '_' + args.train_dir):
        os.makedirs(args.dataset + '_' + args.train_dir)
    timestring = time.strftime('%Y%m%d%H%M%S', time.localtime())
    model_name = 'MAGCL'
    save_dir = os.path.join(args.dataset + '_' + args.train_dir, f'{model_name}_{timestring}')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    f = open(os.path.join(save_dir, f'log_{timestring}.txt'), 'w')
    f.write('\n'.join([str(k) + ':' + str(v) for k, v in vars(args).items()])+'\n')
    f.flush()
    shutil.copy2(sys.argv[0], save_dir)
    shutil.copy2('AGCN.py', save_dir)
    shutil.copy2('model.py', save_dir)
    shutil.copy2('utils.py', save_dir)

    try:
        time_relation_matrix = pickle.load(
            open('data/anchor_relation_time_matrix_%s_%d_%d.pickle' % (args.dataset, args.maxlen, args.time_span), 'rb'))
    except:
        time_relation_matrix = Relation_tem(user_train, usernum, args.maxlen, args.time_span)
        pickle.dump(time_relation_matrix,
                    open('data/anchor_relation_time_matrix_%s_%d_%d.pickle' % (args.dataset, args.maxlen, args.time_span), 'wb'))

    try:
        dis_relation_matrix = pickle.load(
            open('data/anchor_relation_dis_matrix_%s_%d_%d.pickle'%(args.dataset, args.maxlen, args.dis_span),'rb'))
    except:
        dis_relation_matrix = Relation_dis(user_train, usernum, args.maxlen, args.dis_span)
        pickle.dump(dis_relation_matrix,
                    open('data/anchor_relation_dis_matrix_%s_%d_%d.pickle'%(args.dataset, args.maxlen, args.dis_span),'wb'))

    train_dataset = Traindataset(user_train, time_relation_matrix, dis_relation_matrix, itemnum, args.maxlen)
    dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=3) 
    model = MAGCN(usernum, itemnum, timenum, args).to(args.device)

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_uniform_(param.data)
        except:
            pass

    model.train()

    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path,map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)

    ce_criterion = torch.nn.CrossEntropyLoss()
    kl_loss = torch.nn.KLDivLoss(reduction="batchmean")

    weight_decay_list = (param for name, param in model.named_parameters() if name[-4:] != 'bias' and "bn" not in name)
    no_decay_list = (param for name, param in model.named_parameters() if name[-4:] == 'bias' or "bn" in name)
    parameters = [{'params': weight_decay_list},
                  {'params': no_decay_list, 'weight_decay': 0.}]

    adam_optimizer = torch.optim.Adam(parameters, lr=args.lr, betas=(0.9, 0.98),weight_decay=args.l2_emb) 

    T = 0.0
    t0 = time.time()
    anchor_num = args.anchor_num  

    param_count = count_params(model)
    f.write(f'In total: {param_count} trainable parameters.')
    
    # spatial_preference
    spatial_bias = torch.zeros(itemnum+1, itemnum+1)   
    spatial_bias[1:, 1:] = torch.tensor(dis_adj_matrix.A) 
    diag = torch.eye(itemnum+1, dtype=torch.float32)  
    spatial_bias = spatial_bias + diag  # add its spatial proximity of each POI
    spatial_bias[0, 0] = 0.

    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        anchor_idx = torch.randperm(itemnum)[:anchor_num] 
        if not args.same_anchor:
            anchor_idx_set = set(anchor_idx.tolist()) if type(anchor_idx) is np.ndarray else set(anchor_idx.numpy().tolist())
            anchor_idx_tem = np.random.choice(np.array(list(set(range(itemnum)).difference(anchor_idx_set))), anchor_num, replace=False)
            anchor_idx_tem_set = set(anchor_idx_tem.tolist()) if type(anchor_idx_tem) is np.ndarray else set(anchor_idx_tem.numpy().tolist())
            anchor_idx_dis = np.random.choice(np.array(list(set(range(itemnum)).difference(anchor_idx_set.union(anchor_idx_tem_set)))), anchor_num, replace=False)
        else:
            anchor_idx_tem = anchor_idx.clone()
            anchor_idx_dis = anchor_idx.clone()
        
        tra_adj_matrix_anchor = tra_adj_matrix[anchor_idx,:].todense() if type(anchor_idx) is np.ndarray else tra_adj_matrix[anchor_idx.numpy(), :].todense() 
        tra_prior = torch.FloatTensor(tra_adj_matrix_anchor).to(args.device)
        time_adj_matrix_anchor = time_adj_matrix[anchor_idx_tem,:].todense() if type(anchor_idx_tem) is np.ndarray else time_adj_matrix[anchor_idx_tem.numpy(), :].todense() 
        time_prior = torch.FloatTensor(time_adj_matrix_anchor).to(args.device)
        dis_adj_matrix_anchor = dis_adj_matrix[anchor_idx_dis,:].todense() if type(anchor_idx_dis) is np.ndarray else dis_adj_matrix[anchor_idx_dis.numpy(), :].todense() 
        dis_prior = torch.FloatTensor(dis_adj_matrix_anchor).to(args.device)
        
        anchor_idx += 1  # loc_id starts from 1
        anchor_idx_tem += 1
        anchor_idx_dis += 1
        
        anchor_idx = torch.from_numpy(anchor_idx) if type(anchor_idx) is np.ndarray else anchor_idx
        anchor_idx_tem = torch.from_numpy(anchor_idx_tem) if type(anchor_idx_tem) is np.ndarray else anchor_idx_tem
        anchor_idx_dis = torch.from_numpy(anchor_idx_dis) if type(anchor_idx_dis) is np.ndarray else anchor_idx_dis
        
        if args.inference_only: break
        rec_losses, tra_kl_losses, dis_kl_losses, time_kl_losses, contra_losses = [], [], [], [], []
        for step, instance in tqdm(enumerate(dataloader), total=num_batch, ncols=70, leave=False, unit='b'):
            u, seq, time_seq, time_seq_nxt, pos, neg, time_matrix, dis_matrix = instance
            pos_logits, neg_logits, fin_logits, support, support_tem, support_dis, contra_loss = model(u, seq, time_seq, time_seq_nxt, time_matrix, dis_matrix, pos, neg, anchor_idx, anchor_idx_tem, anchor_idx_dis, spatial_bias)
            tra_regular = kl_loss(torch.log(torch.softmax(mask(support.transpose(1,0)), dim=-1) + 1e-9), torch.softmax(mask(tra_prior), dim=-1))
            time_regular = kl_loss(torch.log(torch.softmax(mask(support_tem.transpose(1,0)), dim=-1) + 1e-9), torch.softmax(mask(time_prior), dim=-1))
            dis_regular = kl_loss(torch.log(torch.softmax(mask(support_dis.transpose(1,0)), dim=-1) + 1e-9), torch.softmax(mask(dis_prior), dim=-1))
            
            adam_optimizer.zero_grad()
            pos_label_for_crosse = pos.numpy().reshape(-1)
            indices_for_crosse = np.where(pos_label_for_crosse!=0)[0]
            pos_label_cross = torch.tensor(pos_label_for_crosse[indices_for_crosse], device=args.device)
            loss = ce_criterion(fin_logits[indices_for_crosse], pos_label_cross.long())
            
            rec_losses.append(loss.item())
            tra_kl_losses.append(tra_regular.item())
            time_kl_losses.append(time_regular.item())
            dis_kl_losses.append(dis_regular.item())
            contra_losses.append(contra_loss.item())
            loss += args.tra_kl_reg * tra_regular # default setting from AGRAN
            loss += args.time_kl_reg * time_regular + args.dis_kl_reg * dis_regular + args.contra_reg * contra_loss
            loss.backward()
            adam_optimizer.step()
        print('\nTrain epoch:%d, time: %f(s), rec_loss: %.4f, tra_kl_loss: %.4f, time_kl_loss: %.4f, dis_kl_loss: %.4f, contra_loss: %.4f'
                % (epoch, T, np.array(rec_losses).mean(), np.array(tra_kl_losses).mean(), np.array(time_kl_losses).mean(), np.array(dis_kl_losses).mean(), np.array(contra_losses).mean())) 
        f.write('\nTrain epoch:%d, time: %f(s), rec_loss: %.4f, tra_kl_loss: %.4f, time_kl_loss: %.4f, dis_kl_loss: %.4f, contra_loss: %.4f'
                  % (epoch, T, np.array(rec_losses).mean(), np.array(tra_kl_losses).mean(), np.array(time_kl_losses).mean(), np.array(dis_kl_losses).mean(), np.array(contra_losses).mean())) 
        
        if epoch % args.valid_epoch == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='\n')
            # ### for validation ###
            # NDCG, HR = evaluate_vaild(model, dataset, args, spatial_bias)
            # print('\nValid epoch:%d, time: %f(s), Recall (@2: %.4f, @5: %.4f, @10: %.4f), NDCG (@2: %.4f, @5: %.4f, @10: %.4f)'
            #       % (epoch, T, HR[0], HR[1], HR[2], NDCG[0], NDCG[1], NDCG[2]))
            # f.write('Valid epoch:%d, time: %f(s), Recall (@2: %.4f, @5: %.4f, @10: %.4f), NDCG (@2: %.4f, @5: %.4f, @10: %.4f)'
            #       % (epoch, T, HR[0], HR[1], HR[2], NDCG[0], NDCG[1], NDCG[2]))
            
            ### for test ###
            NDCG, HR, NDCG_user_0, HR_user_0, NDCG_user_1, HR_user_1, NDCG_user_2, HR_user_2, NDCG_user_3, HR_user_3, NDCG_user_4, HR_user_4, NDCG_item_0, HR_item_0, NDCG_item_1, HR_item_1, NDCG_item_2, HR_item_2, NDCG_item_3, HR_item_3, NDCG_item_4, HR_item_4 = evaluate_test(model, dataset, args, spatial_bias)
            print('\nTest epoch:%d, time: %f(s), Recall (@2: %.4f, @5: %.4f, @10: %.4f), NDCG (@2: %.4f, @5: %.4f, @10: %.4f)'
                  % (epoch, T, HR[0], HR[1], HR[2], NDCG[0], NDCG[1], NDCG[2]))
            f.write('\nTest epoch:%d, time: %f(s), Recall (@2: %.4f, @5: %.4f, @10: %.4f), NDCG (@2: %.4f, @5: %.4f, @10: %.4f)\n'
                  % (epoch, T, HR[0], HR[1], HR[2], NDCG[0], NDCG[1], NDCG[2]))

            # print('\n------------User Group-------------------')
            # f.write('\n------------User Group-------------------')
            # print('\nTest epoch:%d, time: %f(s), Recall (@2: %.4f, @5: %.4f, @10: %.4f), NDCG (@2: %.4f, @5: %.4f, @10: %.4f)'
            #       % (epoch, T, HR_user_0[0], HR_user_0[1], HR_user_0[2], NDCG_user_0[0], NDCG_user_0[1], NDCG_user_0[2]))
            # f.write('\nTest epoch:%d, time: %f(s), Recall (@2: %.4f, @5: %.4f, @10: %.4f), NDCG (@2: %.4f, @5: %.4f, @10: %.4f)\n'
            #       % (epoch, T, HR_user_0[0], HR_user_0[1], HR_user_0[2], NDCG_user_0[0], NDCG_user_0[1], NDCG_user_0[2]))
            
            # print('\nTest epoch:%d, time: %f(s), Recall (@2: %.4f, @5: %.4f, @10: %.4f), NDCG (@2: %.4f, @5: %.4f, @10: %.4f)'
            #       % (epoch, T, HR_user_1[0], HR_user_1[1], HR_user_1[2], NDCG_user_1[0], NDCG_user_1[1], NDCG_user_1[2]))
            # f.write('\nTest epoch:%d, time: %f(s), Recall (@2: %.4f, @5: %.4f, @10: %.4f), NDCG (@2: %.4f, @5: %.4f, @10: %.4f)\n'
            #       % (epoch, T, HR_user_1[0], HR_user_1[1], HR_user_1[2], NDCG_user_1[0], NDCG_user_1[1], NDCG_user_1[2]))
            
            # print('\nTest epoch:%d, time: %f(s), Recall (@2: %.4f, @5: %.4f, @10: %.4f), NDCG (@2: %.4f, @5: %.4f, @10: %.4f)'
            #       % (epoch, T, HR_user_2[0], HR_user_2[1], HR_user_2[2], NDCG_user_2[0], NDCG_user_2[1], NDCG_user_2[2]))
            # f.write('\nTest epoch:%d, time: %f(s), Recall (@2: %.4f, @5: %.4f, @10: %.4f), NDCG (@2: %.4f, @5: %.4f, @10: %.4f)\n'
            #       % (epoch, T, HR_user_2[0], HR_user_2[1], HR_user_2[2], NDCG_user_2[0], NDCG_user_2[1], NDCG_user_2[2]))
            
            # print('\nTest epoch:%d, time: %f(s), Recall (@2: %.4f, @5: %.4f, @10: %.4f), NDCG (@2: %.4f, @5: %.4f, @10: %.4f)'
            #       % (epoch, T, HR_user_3[0], HR_user_3[1], HR_user_3[2], NDCG_user_3[0], NDCG_user_3[1], NDCG_user_3[2]))
            # f.write('\nTest epoch:%d, time: %f(s), Recall (@2: %.4f, @5: %.4f, @10: %.4f), NDCG (@2: %.4f, @5: %.4f, @10: %.4f)\n'
            #       % (epoch, T, HR_user_3[0], HR_user_3[1], HR_user_3[2], NDCG_user_3[0], NDCG_user_3[1], NDCG_user_3[2]))
            
            # print('\nTest epoch:%d, time: %f(s), Recall (@2: %.4f, @5: %.4f, @10: %.4f), NDCG (@2: %.4f, @5: %.4f, @10: %.4f)'
            #       % (epoch, T, HR_user_4[0], HR_user_4[1], HR_user_4[2], NDCG_user_4[0], NDCG_user_4[1], NDCG_user_4[2]))
            # f.write('\nTest epoch:%d, time: %f(s), Recall (@2: %.4f, @5: %.4f, @10: %.4f), NDCG (@2: %.4f, @5: %.4f, @10: %.4f)\n'
            #       % (epoch, T, HR_user_4[0], HR_user_4[1], HR_user_4[2], NDCG_user_4[0], NDCG_user_4[1], NDCG_user_4[2]))
            
            # print('\n------------Item Group-------------------')
            # f.write('\n------------Item Group-------------------')
            # print('\nTest epoch:%d, time: %f(s), Recall (@2: %.4f, @5: %.4f, @10: %.4f), NDCG (@2: %.4f, @5: %.4f, @10: %.4f)'
            #       % (epoch, T, HR_item_0[0], HR_item_0[1], HR_item_0[2], NDCG_item_0[0], NDCG_item_0[1], NDCG_item_0[2]))
            # f.write('\nTest epoch:%d, time: %f(s), Recall (@2: %.4f, @5: %.4f, @10: %.4f), NDCG (@2: %.4f, @5: %.4f, @10: %.4f)\n'
            #       % (epoch, T, HR_item_0[0], HR_item_0[1], HR_item_0[2], NDCG_item_0[0], NDCG_item_0[1], NDCG_item_0[2]))
            
            # print('\nTest epoch:%d, time: %f(s), Recall (@2: %.4f, @5: %.4f, @10: %.4f), NDCG (@2: %.4f, @5: %.4f, @10: %.4f)'
            #       % (epoch, T, HR_item_1[0], HR_item_1[1], HR_item_1[2], NDCG_item_1[0], NDCG_item_1[1], NDCG_item_1[2]))
            # f.write('\nTest epoch:%d, time: %f(s), Recall (@2: %.4f, @5: %.4f, @10: %.4f), NDCG (@2: %.4f, @5: %.4f, @10: %.4f)\n'
            #       % (epoch, T, HR_item_1[0], HR_item_1[1], HR_item_1[2], NDCG_item_1[0], NDCG_item_1[1], NDCG_item_1[2]))
            
            # print('\nTest epoch:%d, time: %f(s), Recall (@2: %.4f, @5: %.4f, @10: %.4f), NDCG (@2: %.4f, @5: %.4f, @10: %.4f)'
            #       % (epoch, T, HR_item_2[0], HR_item_2[1], HR_item_2[2], NDCG_item_2[0], NDCG_item_2[1], NDCG_item_2[2]))
            # f.write('\nTest epoch:%d, time: %f(s), Recall (@2: %.4f, @5: %.4f, @10: %.4f), NDCG (@2: %.4f, @5: %.4f, @10: %.4f)\n'
            #       % (epoch, T, HR_item_2[0], HR_item_2[1], HR_item_2[2], NDCG_item_2[0], NDCG_item_2[1], NDCG_item_2[2]))
            
            # print('\nTest epoch:%d, time: %f(s), Recall (@2: %.4f, @5: %.4f, @10: %.4f), NDCG (@2: %.4f, @5: %.4f, @10: %.4f)'
            #       % (epoch, T, HR_item_3[0], HR_item_3[1], HR_item_3[2], NDCG_item_3[0], NDCG_item_3[1], NDCG_item_3[2]))
            # f.write('\nTest epoch:%d, time: %f(s), Recall (@2: %.4f, @5: %.4f, @10: %.4f), NDCG (@2: %.4f, @5: %.4f, @10: %.4f)\n'
            #       % (epoch, T, HR_item_3[0], HR_item_3[1], HR_item_3[2], NDCG_item_3[0], NDCG_item_3[1], NDCG_item_3[2]))
            
            # print('\nTest epoch:%d, time: %f(s), Recall (@2: %.4f, @5: %.4f, @10: %.4f), NDCG (@2: %.4f, @5: %.4f, @10: %.4f)'
            #       % (epoch, T, HR_item_4[0], HR_item_4[1], HR_item_4[2], NDCG_item_4[0], NDCG_item_4[1], NDCG_item_4[2]))
            # f.write('\nTest epoch:%d, time: %f(s), Recall (@2: %.4f, @5: %.4f, @10: %.4f), NDCG (@2: %.4f, @5: %.4f, @10: %.4f)\n'
            #       % (epoch, T, HR_item_4[0], HR_item_4[1], HR_item_4[2], NDCG_item_4[0], NDCG_item_4[1], NDCG_item_4[2]))
            
            f.flush()
            t0 = time.time()
            model.train()

    f.close()
    print("Done")
