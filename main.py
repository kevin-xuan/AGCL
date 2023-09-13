import os
import time
import torch
import pickle
import argparse
from dataset import Traindataset
from torch.utils.data import DataLoader
import scipy.sparse as sp
from model_ag import AGRAN
from tqdm import tqdm
from utils_ag import *

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
parser.add_argument('--dataset', default='four-sin')
parser.add_argument('--train_dir', default='default')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--maxlen', default=50, type=int, help="use recent maxlen subsequence to train model")
parser.add_argument('--hidden_units', default=64, type=int, help="embedding dimension")
parser.add_argument('--num_blocks', default=3, type=int, help="number of transformer layer")
parser.add_argument('--num_epochs', default=121, type=int)
parser.add_argument('--num_heads', default=2, type=int)
parser.add_argument('--dropout_rate', default=0.3, type=float)
parser.add_argument('--l2_emb', default=0.0001, type=float, help="L2 regularization")
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str, help="model checkpoint")
parser.add_argument('--time_span', default=256, type=int, help="maximal time interval threshold")
parser.add_argument('--dis_span', default=256, type=int, help="maximal distance interval threshold")
parser.add_argument('--kl_reg', default=1.0, type=float)
parser.add_argument('--layer_num', default=4, type=int, help="number of GCN layers")
parser.add_argument('--valid_epoch', default=2, type=int, help="every n epoch to test")

args = parser.parse_args()
if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)


def mask(adj, epsilon=0, mask_value=-1e16):
    mask = (adj > epsilon).detach().float()
    update_adj = adj * mask + (1 - mask) * mask_value
    return update_adj

if __name__ == '__main__':
    args.device = torch.device("cuda:{}".format(args.gpu)) if torch.cuda.is_available() else torch.device("cpu")
    dataset = data_partition(args.dataset)
    [user_train, user_valid, user_test, usernum, itemnum, timenum] = dataset  # timenum: maximal scaled time slot among all users' check-ins
    num_batch = len(user_train) // args.batch_size  # 4368 // 128 = 36 #* how to deal with remaining part?

    # tra_adj_matrix = sp.load_npz('data/sin_transaction_kl_notest.npz')  # N x N, frequency-based transition graph
    tra_adj_matrix = sp.load_npz('data/%s_transaction_kl_notest.npz' % args.dataset)  # csr_matrix
    prior = torch.FloatTensor(tra_adj_matrix.todense()).to(args.device)  # Equation (11)

    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))  # Tabel 1
    print(itemnum)
    print(usernum)

    timestring = time.strftime('%Y%m%d%H%M%S', time.localtime())
    f = open(os.path.join(args.dataset + '_' + args.train_dir, f'log_{timestring}.txt'), 'w')
    f.write('\n'.join([str(k) + ':' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])])+'\n')
    f.flush()

    try:
        time_relation_matrix = pickle.load(
            open('data/relation_time_matrix_%s_%d_%d.pickle' % (args.dataset, args.maxlen, args.time_span), 'rb'))
    except:
        time_relation_matrix = Relation_tem(user_train, usernum, args.maxlen, args.time_span)  # compute the time distance between two different items in train sequence, i.e., equation (12)
        pickle.dump(time_relation_matrix,
                    open('data/relation_time_matrix_%s_%d_%d.pickle' % (args.dataset, args.maxlen, args.time_span), 'wb'))

    try:
        dis_relation_matrix = pickle.load(
            open('data/relation_dis_matrix_%s_%d_%d.pickle'%(args.dataset, args.maxlen, args.dis_span),'rb'))
    except:
        dis_relation_matrix = Relation_dis(user_train, usernum, args.maxlen, args.dis_span)  # compute the spatial distance between two different items in train sequence, i.e., equation (13)
        pickle.dump(dis_relation_matrix,
                    open('data/relation_dis_matrix_%s_%d_%d.pickle'%(args.dataset, args.maxlen, args.dis_span),'wb'))  

    train_dataset = Traindataset(user_train, time_relation_matrix, dis_relation_matrix, itemnum, args.maxlen)
    dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=3)

    model = AGRAN(usernum, itemnum, itemnum, args).to(args.device)

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_uniform_(param.data)
        except:
            pass

    model.train()

    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)


    ce_criterion = torch.nn.CrossEntropyLoss()
    kl_loss = torch.nn.KLDivLoss(reduction="batchmean")

    weight_decay_list = (param for name, param in model.named_parameters() if name[-4:] != 'bias' and "bn" not in name)  # 1e-4 regularization
    no_decay_list = (param for name, param in model.named_parameters() if name[-4:] == 'bias' or "bn" in name)
    parameters = [{'params': weight_decay_list},
                  {'params': no_decay_list, 'weight_decay': 0.}]

    adam_optimizer = torch.optim.Adam(parameters, lr=args.lr, betas=(0.9, 0.98), weight_decay=args.l2_emb)  # 1e-4
    

    T = 0.0
    t0 = time.time()

    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only: break
        for step, instance in tqdm(enumerate(dataloader), total=num_batch, ncols=70, leave=False, unit='b'):
            u, seq, time_seq, pos, neg, time_matrix, dis_matrix  = instance
            pos_logits, neg_logits, fin_logits, padding, support = model(u, seq, time_matrix, dis_matrix, pos, neg)
            a = kl_loss(torch.log(torch.softmax(mask(support),dim=-1)+1e-9), torch.softmax(mask(prior),dim=-1))  # Equation (11)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape,
                                                                                                   device=args.device)  # (B, N)
            adam_optimizer.zero_grad()
            # indices = np.where(pos != 0)  # labels, 0 is padding
            pos_label_for_crosse = pos.numpy().reshape(-1)  # (B * N, )


            indices_for_crosse = np.where(pos_label_for_crosse!=0)

            pos_label_cross = torch.tensor(pos_label_for_crosse[indices_for_crosse], device=args.device)
            loss = ce_criterion(fin_logits[indices_for_crosse], pos_label_cross.long())  #* Equation (19) neg_logits is useless here
            kl_reg = args.kl_reg  # 1.
            if epoch >= 0:
                loss +=  kl_reg * a

            loss.backward()
            adam_optimizer.step()

        if  epoch % args.valid_epoch == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='\n')
            ### for validation ###
            NDCG, HR = evaluate_vaild(model, dataset, args)
            print('\nValid epoch:%d, time: %f(s), Recall_val (@2: %.4f, @5: %.4f, @10: %.4f), NDCG_val (@2: %.4f, @5: %.4f, @10: %.4f)'
                  % (epoch, T, HR[0], HR[1], HR[2], NDCG[0], NDCG[1], NDCG[2]))
            f.write('\nValid epoch:%d, time: %f(s), Recall (@2: %.4f, @5: %.4f, @10: %.4f), NDCG (@2: %.4f, @5: %.4f, @10: %.4f)'
                  % (epoch, T, HR[0], HR[1], HR[2], NDCG[0], NDCG[1], NDCG[2]))
            
            ### for test ###
            NDCG,HR = evaluate_test(model, dataset, args)
            print('\nTest epoch:%d, time: %f(s), Recall (@2: %.4f, @5: %.4f, @10: %.4f), NDCG (@2: %.4f, @5: %.4f, @10: %.4f)'
                  % (epoch, T, HR[0], HR[1], HR[2], NDCG[0], NDCG[1], NDCG[2]))
            f.write('\nTest epoch:%d, time: %f(s), Recall (@2: %.4f, @5: %.4f, @10: %.4f), NDCG (@2: %.4f, @5: %.4f, @10: %.4f)\n'
                  % (epoch, T, HR[0], HR[1], HR[2], NDCG[0], NDCG[1], NDCG[2]))
            f.flush()
            t0 = time.time()
            model.train()
    
    f.close()
    print("Done")
