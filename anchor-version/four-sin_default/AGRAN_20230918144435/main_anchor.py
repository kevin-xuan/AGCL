import os
import time
import torch
import pickle
import argparse
import scipy.sparse as sp
from dataset import Traindataset
from torch.utils.data import DataLoader
from model_ag import AGRAN_anchor
from tqdm import tqdm
from utils_ag import *
import random
import faiss, shutil

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

def select_anchor(kmeans, index, item_embs):
    faiss_item_embs = item_embs[1:, ].clone().detach().cpu()
    faiss_item_embs = faiss_item_embs.div(torch.norm(faiss_item_embs, p=2, dim=-1, keepdim=True)).numpy()
    kmeans.train(faiss_item_embs)
    index.add(faiss_item_embs)
    D, I = index.search(kmeans.centroids, 1)  # find the top-1 nearest point in item_embs to the centroids to select r anchors
    anchor_idx = torch.from_numpy(I).squeeze(-1)
    return anchor_idx

def sort_by_importance(mx):
    '''Column-wise sum'''
    colsum = -np.array(mx.sum(0))  # (N, )
    index = np.argsort(colsum)
    return index  # (r, )

def setup_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # 确定性固定
    torch.backends.cudnn.benchmark = True  # False会确定性地选择算法，会降低性能
    torch.backends.cudnn.enabled = True   # 增加运行效率，默认就是True

setup_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='four-sin',type=str)  # 'gowalla'  
parser.add_argument('--train_dir', default='default',type=str)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int, help="use recent maxlen subsequence to train model")
parser.add_argument('--hidden_units', default=64, type=int, help="embedding dimension")
parser.add_argument('--num_blocks', default=2, type=int, help="number of transformer layer")
parser.add_argument('--num_epochs', default=50, type=int)  # 200  70
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--l2_emb', default=0.001, type=float, help="L2 regularization")
parser.add_argument('--gpu', default=1, type=int)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str, help="model checkpoint")
parser.add_argument('--time_span', default=128, type=int, help="maximal time interval threshold")  # 256
parser.add_argument('--dis_span', default=256, type=int, help="maximal distance interval threshold")  # 256
parser.add_argument('--anchor_num', default=500, type=int)  
parser.add_argument('--layer_num', default=3, type=int, help="number of GCN layers")

parser.add_argument('--tra_kl_reg', default=1.0, type=float)
parser.add_argument('--dis_kl_reg', default=1.0, type=float)  # 0.0
parser.add_argument('--time_kl_reg', default=1.0, type=float)
parser.add_argument('--contra_reg', default=1.0, type=float)
parser.add_argument('--tra_delta', default=0., type=float)
parser.add_argument('--tem_delta', default=0., type=float)
parser.add_argument('--dis_delta', default=0., type=float)
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
    # tra_adj_matrix = sp.load_npz('../data/sin_transaction_kl_notest.npz')  # csr_matrix
    tra_adj_matrix = sp.load_npz('../data/%s_transaction_kl_notest.npz' % args.dataset)  # N x N
    
    # Graph-Flashback
    with open('../KGE/POI_graph/foursquare_scheme2_transe_loc_spatial_20.pkl', 'rb') as f:  
        dis_adj_matrix = pickle.load(f)  
    with open('../KGE/POI_graph/foursquare_scheme2_transe_loc_temporal_20.pkl', 'rb') as f:  
        time_adj_matrix = pickle.load(f)  # 在cpu上
    tra_adj_matrix = tra_adj_matrix.todok()
    dis_adj_matrix = dis_adj_matrix.todok()
    time_adj_matrix = time_adj_matrix.todok()
    interaction_matrix = sp.load_npz('../data/%s_interaction_kl_notest.npz' % args.dataset)  # M x N better than graph-flashback interaction matrix
    # with open('../KGE/POI_graph/foursquare_scheme2_transe_user-loc_20.pkl', 'rb') as f:  
    #     interaction_matrix = pickle.load(f) 
    
    args.device = torch.device("cuda:{}".format(args.gpu)) if torch.cuda.is_available() else torch.device("cpu")
    # args.device = torch.device("cpu")

    dataset = data_partition(args.dataset)
    [user_train, user_valid, user_test, usernum, itemnum, timenum] = dataset  # timenum is useless
    num_batch = len(user_train) // args.batch_size
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print(itemnum)
    print(usernum)
    print('average sequence length: %.2f' % (cc / len(user_train)))

    if not os.path.isdir(args.dataset + '_' + args.train_dir):
        os.makedirs(args.dataset + '_' + args.train_dir)
    timestring = time.strftime('%Y%m%d%H%M%S', time.localtime())
    model_name = 'AGRAN'
    save_dir = os.path.join(args.dataset + '_' + args.train_dir, f'{model_name}_{timestring}')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    f = open(os.path.join(save_dir, f'log_{timestring}.txt'), 'w')
    f.write('\n'.join([str(k) + ':' + str(v) for k, v in vars(args).items()])+'\n')
    f.flush()
    shutil.copy2(sys.argv[0], save_dir)
    shutil.copy2('AGCN.py', save_dir)
    shutil.copy2('model_ag.py', save_dir)
    shutil.copy2('utils_ag.py', save_dir)

    try:
        time_relation_matrix = pickle.load(
            open('../data/anchor_relation_time_matrix_%s_%d_%d.pickle' % (args.dataset, args.maxlen, args.time_span), 'rb'))
    except:
        time_relation_matrix = Relation_tem(user_train, usernum, args.maxlen, args.time_span)
        pickle.dump(time_relation_matrix,
                    open('../data/anchor_relation_time_matrix_%s_%d_%d.pickle' % (args.dataset, args.maxlen, args.time_span), 'wb'))

    try:
        dis_relation_matrix = pickle.load(
            open('../data/anchor_relation_dis_matrix_%s_%d_%d.pickle'%(args.dataset, args.maxlen, args.dis_span),'rb'))
    except:
        dis_relation_matrix = Relation_dis(user_train, usernum, args.maxlen, args.dis_span)
        pickle.dump(dis_relation_matrix,
                    open('../data/anchor_relation_dis_matrix_%s_%d_%d.pickle'%(args.dataset, args.maxlen, args.dis_span),'wb'))

    train_dataset = Traindataset(user_train, time_relation_matrix, dis_relation_matrix, itemnum, args.maxlen)
    dataloader = DataLoader(dataset=train_dataset,batch_size=args.batch_size, shuffle=True, num_workers=3) 
    model = AGRAN_anchor(usernum, itemnum, timenum, args).to(args.device)

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

    adam_optimizer = torch.optim.Adam(parameters, lr=args.lr, betas=(0.9, 0.98),weight_decay=args.l2_emb)  # 1e-3

    T = 0.0
    t0 = time.time()
    anchor_num = args.anchor_num  # r = 500
    # delta = 0.02 # 1.0
    # anchor_idx = None
    # indices = sort_by_importance(interaction_matrix.A)
    # static_anchor_idx = indices[:int(round(anchor_num * delta))]
    # sample_range = set(range(itemnum)).difference(set(static_anchor_idx.tolist()))
    
    # f.write('\nTrainable parameter list:')
    param_count = count_params(model)
    f.write(f'In total: {param_count} trainable parameters.')
    
    #* spatial_preference
    # lambda_dis = 1.
    # I = identity(dis_adj_matrix.shape[0], format='coo')
    # dis_graph = coo_matrix(dis_adj_matrix)
    # dis_graph = sparse_matrix_to_tensor(calculate_random_walk_matrix((dis_graph * lambda_dis + I).astype(np.float32))).to(args.device)
    spatial_bias = torch.zeros(itemnum+1, itemnum+1)   # (N_all, N_all)
    spatial_bias[1:, 1:] = torch.tensor(dis_adj_matrix.A)
    diag = torch.eye(itemnum+1, dtype=torch.float32)  
    spatial_bias = spatial_bias + diag  # POI与本身的空间距离关系可以设置为一个超参数,这里先用1
    spatial_bias[0, 0] = 0.
    #* user_preference
    user_bias = torch.zeros(usernum+1, itemnum+1).to(args.device)   # (M_all, N_all)
    user_bias[1:, 1:] = torch.tensor(interaction_matrix.A)

    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        anchor_idx = torch.randperm(itemnum)[:anchor_num] # (r, )
        
        # dynamic_anchor_idx = np.random.choice(np.array(list(sample_range)), anchor_num - len(static_anchor_idx), replace=False)
        # anchor_idx = np.concatenate([static_anchor_idx, dynamic_anchor_idx], axis=0)
        
        anchor_idx_set = set(anchor_idx.tolist()) if type(anchor_idx) is np.ndarray else set(anchor_idx.numpy().tolist())
        
        anchor_idx_tem = np.random.choice(np.array(list(set(range(itemnum)).difference(anchor_idx_set))), anchor_num, replace=False)
        
        anchor_idx_tem_set = set(anchor_idx_tem.tolist()) if type(anchor_idx_tem) is np.ndarray else set(anchor_idx_tem.numpy().tolist())
        
        anchor_idx_dis = np.random.choice(np.array(list(set(range(itemnum)).difference(anchor_idx_set.union(anchor_idx_tem_set)))), anchor_num, replace=False)
        # anchor_idx_dis = anchor_idx_tem  #* same anchor_idx as anchor_idx_tem
        # dynamic_anchor_idx_tem = np.random.choice(np.array(list(sample_range.difference(anchor_idx_set))), anchor_num - len(static_anchor_idx), replace=False)
        # anchor_idx_tem = np.concatenate([static_anchor_idx, dynamic_anchor_idx_tem], axis=0)
        
        
        tra_adj_matrix_anchor = tra_adj_matrix[anchor_idx,:].todense() if type(anchor_idx) is np.ndarray else tra_adj_matrix[anchor_idx.numpy(), :].todense() # (r, N)
        tra_prior = torch.FloatTensor(tra_adj_matrix_anchor).to(args.device)
        dis_adj_matrix_anchor = dis_adj_matrix[anchor_idx_dis,:].todense() if type(anchor_idx_dis) is np.ndarray else dis_adj_matrix[anchor_idx_dis.numpy(), :].todense() # (r, N)
        dis_prior = torch.FloatTensor(dis_adj_matrix_anchor).to(args.device)
        time_adj_matrix_anchor = time_adj_matrix[anchor_idx_tem,:].todense() if type(anchor_idx_tem) is np.ndarray else time_adj_matrix[anchor_idx_tem.numpy(), :].todense() # (r, N)
        time_prior = torch.FloatTensor(time_adj_matrix_anchor).to(args.device)
        
        anchor_idx += 1  # loc_id starts from 1
        anchor_idx_tem += 1
        anchor_idx_dis += 1
        anchor_idx = torch.from_numpy(anchor_idx) if type(anchor_idx) is np.ndarray else anchor_idx
        anchor_idx_tem = torch.from_numpy(anchor_idx_tem) if type(anchor_idx_tem) is np.ndarray else anchor_idx_tem
        anchor_idx_dis = torch.from_numpy(anchor_idx_dis) if type(anchor_idx_dis) is np.ndarray else anchor_idx_dis
        if args.inference_only: break
        # #* clustering at each epoch   
        # index = faiss.IndexFlatL2(args.hidden_units)
        # kmeans = faiss.Kmeans(args.hidden_units, args.anchor_num, niter=5, verbose=False)
        rec_losses, tra_kl_losses, dis_kl_losses, time_kl_losses = [], [], [], []
        contra_losses = []
        for step, instance in tqdm(enumerate(dataloader), total=num_batch, ncols=70, leave=False, unit='b'):
            u, seq, time_seq, pos, neg, time_matrix, dis_matrix = instance
            item_embs, pos_logits, neg_logits, fin_logits, support, support_tem, support_dis, contra_loss = model(u, seq, time_matrix, dis_matrix, pos, neg, anchor_idx, anchor_idx_tem, anchor_idx_dis, time_adj_matrix, dis_adj_matrix, spatial_bias, user_bias)
            tra_regular = kl_loss(torch.log(torch.softmax(mask(support.transpose(1,0)),dim=-1)+1e-9),torch.softmax(mask(tra_prior),dim=-1))
            time_regular = kl_loss(torch.log(torch.softmax(mask(support_tem.transpose(1,0)),dim=-1)+1e-9),torch.softmax(mask(time_prior),dim=-1))
            dis_regular = kl_loss(torch.log(torch.softmax(mask(support_dis.transpose(1,0)),dim=-1)+1e-9),torch.softmax(mask(dis_prior),dim=-1))
    
            adam_optimizer.zero_grad()
            pos_label_for_crosse = pos.numpy().reshape(-1)

            indices_for_crosse = np.where(pos_label_for_crosse!=0)

            pos_label_cross = torch.tensor(pos_label_for_crosse[indices_for_crosse], device=args.device)
            loss = ce_criterion(fin_logits[indices_for_crosse], pos_label_cross.long())
            rec_losses.append(loss.item())
            tra_kl_losses.append(tra_regular.item())
            time_kl_losses.append(time_regular.item())
            dis_kl_losses.append(dis_regular.item())
            contra_losses.append(contra_loss.item())
            loss += args.tra_kl_reg * tra_regular + args.time_kl_reg * time_regular #+ args.dis_kl_reg * dis_regular 
            loss += args.dis_kl_reg * dis_regular
            # loss += args.contra_reg * contra_loss
            loss.backward()
            adam_optimizer.step()
        print('\nTrain epoch:%d, time: %f(s), rec_loss: %.4f, tra_kl_loss: %.4f, time_kl_loss: %.4f, dis_kl_loss: %.4f, contra_loss: %.4f'
                % (epoch, T, np.array(rec_losses).mean(), np.array(tra_kl_losses).mean(), np.array(time_kl_losses).mean(), np.array(dis_kl_losses).mean(), np.array(contra_losses).mean())) 
        f.write('\nTrain epoch:%d, time: %f(s), rec_loss: %.4f, tra_kl_loss: %.4f, time_kl_loss: %.4f, dis_kl_loss: %.4f, contra_loss: %.4f'
                  % (epoch, T, np.array(rec_losses).mean(), np.array(tra_kl_losses).mean(), np.array(time_kl_losses).mean(), np.array(dis_kl_losses).mean(), np.array(contra_losses).mean())) 
        
        # anchor_idx = select_anchor(kmeans, index, item_embs)
        if epoch % args.valid_epoch == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='\n')
            # ### for validation ###
            # NDCG, HR = evaluate_vaild(model, dataset, args, time_adj_matrix, dis_adj_matrix)
            # print('\nValid epoch:%d, time: %f(s), Recall (@2: %.4f, @5: %.4f, @10: %.4f), NDCG (@2: %.4f, @5: %.4f, @10: %.4f)'
            #       % (epoch, T, HR[0], HR[1], HR[2], NDCG[0], NDCG[1], NDCG[2]))
            # f.write('Valid epoch:%d, time: %f(s), Recall (@2: %.4f, @5: %.4f, @10: %.4f), NDCG (@2: %.4f, @5: %.4f, @10: %.4f)'
            #       % (epoch, T, HR[0], HR[1], HR[2], NDCG[0], NDCG[1], NDCG[2]))
            
            ### for test ###
            NDCG, HR = evaluate_test(model, dataset, args, time_adj_matrix, dis_adj_matrix, spatial_bias, user_bias)
            print('\nTest epoch:%d, time: %f(s), Recall (@2: %.4f, @5: %.4f, @10: %.4f), NDCG (@2: %.4f, @5: %.4f, @10: %.4f)'
                  % (epoch, T, HR[0], HR[1], HR[2], NDCG[0], NDCG[1], NDCG[2]))
            f.write('\nTest epoch:%d, time: %f(s), Recall (@2: %.4f, @5: %.4f, @10: %.4f), NDCG (@2: %.4f, @5: %.4f, @10: %.4f)\n'
                  % (epoch, T, HR[0], HR[1], HR[2], NDCG[0], NDCG[1], NDCG[2]))

            f.flush()
            t0 = time.time()
            model.train()

    f.close()
    print("Done")
