import torch
import torch.nn.functional as F
import torch.nn as nn
from scipy.sparse import coo_matrix, identity
import numpy as np
import scipy.sparse as sp

class AGCN_anchor(nn.Module):
    def __init__(self,
                 args,
                 input_dim,
                 output_dim,
                 layer,
                 bias=False):

        super(AGCN_anchor,self).__init__()
        self.tra_delta = args.tra_delta
        self.tem_delta = args.tem_delta
        self.dis_delta = args.dis_delta
        self.layer_num = layer  # 3
        self.output_dim = output_dim
        self.cos_weight_tra = nn.Linear(input_dim, output_dim)
        self.cos_weight_tem = nn.Linear(input_dim, output_dim)
        self.cos_weight_dis = nn.Linear(input_dim, output_dim)
        
        # self.lambda_tem = 1.0
        # self.lambda_dis = 1.0
    
    def cosine_matrix_div(self, emb, anchor, type='tra'):
        if type == 'tra':
            emb = self.cos_weight_tra(emb)
            anchor = self.cos_weight_tra(anchor)
        elif type == 'tem':
            emb = self.cos_weight_tem(emb)
            anchor = self.cos_weight_tem(anchor)
        elif type == 'dis':
            emb = self.cos_weight_dis(emb)
            anchor = self.cos_weight_dis(anchor)
        else:
            raise NotImplementedError
        
        node_norm = emb.div(torch.norm(emb, p=2, dim=-1, keepdim=True))
        anchor_norm = anchor.div(torch.norm(anchor, p=2, dim=-1, keepdim=True))
        cos_adj = torch.mm(node_norm, anchor_norm.transpose(-1, -2))  # (N, r)
        
        return cos_adj
    
    def get_neighbor_hard_threshold(self, adj, epsilon=0, mask_value=0):
        mask = (adj > epsilon).detach().float()
        update_adj = adj * mask + (1 - mask) * mask_value
        return update_adj
    
    def predefined_embedding(self, emb, tem_graph, dis_graph):
        x_fin_tem, x_fin_dis = [emb], [emb]
        layer_tem, layer_dis = emb, emb
        I = identity(tem_graph.shape[0], format='coo')
        tem_graph, dis_graph = coo_matrix(tem_graph), coo_matrix(dis_graph)
        tem_graph = sparse_matrix_to_tensor(calculate_random_walk_matrix((tem_graph * self.lambda_tem + I).astype(np.float32))).to(emb.device)
        dis_graph = sparse_matrix_to_tensor(calculate_random_walk_matrix((dis_graph * self.lambda_dis + I).astype(np.float32))).to(emb.device)
        for f in range(self.layer_num): 
            layer_tem = torch.sparse.mm(tem_graph, layer_tem)
            layer_dis = torch.sparse.mm(dis_graph, layer_dis)
            x_fin_tem += [layer_tem]
            x_fin_dis += [layer_dis]
        x_fin_tem = torch.stack(x_fin_tem, dim=1)  
        x_fin_dis = torch.stack(x_fin_dis, dim=1)  
        output_tem = torch.sum(x_fin_tem, dim=1)  # (N, D)
        output_dis = torch.sum(x_fin_dis, dim=1)  # (N, D)

        return output_tem, output_dis

    def gcn(self, x, graph):
        x_fin = [x]
        layer = x
        for f in range(self.layer_num):  # 3 GCN
            node_norm = graph / torch.clamp(torch.sum(graph, dim=-2, keepdim=True), min=1e-12) # Equation (5)
            anchor_norm = graph / torch.clamp(torch.sum(graph, dim=-1,keepdim=True), min=1e-12)
            layer = torch.matmul(anchor_norm, torch.matmul(node_norm.transpose(-1,-2), layer)) + layer  # Equation (9)
            x_fin += [layer]
        x_fin = torch.stack(x_fin, dim=1)  
        output = torch.sum(x_fin, dim=1)  # (N, D)
        return output
    
    def forward(self, inputs, anchor_idx, anchor_idx_tem, anchor_idx_dis, time_adj_matrix=None, dis_adj_matrix=None):
        x = inputs.weight[1:,:]  # (N, D)
        anchor = inputs(anchor_idx)  # (r, D)
        anchor_tem = inputs(anchor_idx_tem)  # (r, D)
        anchor_dis = inputs(anchor_idx_dis)  # (r, D)
        anchor_adj_tra = self.cosine_matrix_div(x, anchor, type='tra')
        anchor_adj_tem = self.cosine_matrix_div(x, anchor_tem, type='tem')
        anchor_adj_dis = self.cosine_matrix_div(x, anchor_dis, type='dis')
        anchor_adj_tra = self.get_neighbor_hard_threshold(anchor_adj_tra, epsilon=self.tra_delta)  # (N, r)
        anchor_adj_tem = self.get_neighbor_hard_threshold(anchor_adj_tem, epsilon=self.tem_delta)  # (N, r)
        anchor_adj_dis = self.get_neighbor_hard_threshold(anchor_adj_dis, epsilon=self.dis_delta)  # (N, r)

        output_tra = self.gcn(x, anchor_adj_tra)
        output_tem = self.gcn(x, anchor_adj_tem)
        output_dis = self.gcn(x, anchor_adj_dis)
        mp_tra = torch.cat([inputs.weight[0, :].unsqueeze(dim=0), output_tra], dim=0)
        mp_tem = torch.cat([inputs.weight[0, :].unsqueeze(dim=0), output_tem], dim=0)
        mp_dis = torch.cat([inputs.weight[0, :].unsqueeze(dim=0), output_dis], dim=0)
        
        return mp_tra, mp_tem, mp_dis, anchor_adj_tra, anchor_adj_tem, anchor_adj_dis, anchor_adj_tra.argmax(-1), anchor_adj_tem.argmax(-1), anchor_adj_dis.argmax(-1)
        return mp_tra, mp_tem, mp_dis, anchor_adj_tra, anchor_adj_tem, anchor_adj_dis
        # if time_adj_matrix is not None:
        #     output_tem, output_dis = self.predefined_embedding(x, time_adj_matrix, dis_adj_matrix)
        #     mp_tem = torch.cat([inputs.weight[0, :].unsqueeze(dim=0), output_tem], dim=0)
        #     mp_dis = torch.cat([inputs.weight[0, :].unsqueeze(dim=0), output_dis], dim=0)
        # else:
        #     mp_tem, mp_dis = None, None
        # return mp2, anchor_adj_tra, anchor_adj_tem, anchor_adj_dis

def sparse_matrix_to_tensor(graph):
    graph = coo_matrix(graph)
    vaules = graph.data
    indices = np.vstack((graph.row, graph.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(vaules)
    shape = graph.shape
    graph = torch.sparse_coo_tensor(i, v, torch.Size(shape))

    return graph


def calculate_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()

    return random_walk_mx  # D^-1 W