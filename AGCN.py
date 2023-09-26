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
                 layer
                ):

        super(AGCN_anchor,self).__init__()
        self.tra_delta = args.tra_delta  # mask threshold for adaptive weights
        self.tem_delta = args.tem_delta
        self.dis_delta = args.dis_delta
        self.layer_num = layer  # 3
        self.output_dim = output_dim
        self.args = args
        self.cos_weight_tra = nn.Linear(input_dim, output_dim)
        if args.time_kl_reg > 0:
            self.cos_weight_tem = nn.Linear(input_dim, output_dim)
        if args.dis_kl_reg > 0:
            self.cos_weight_dis = nn.Linear(input_dim, output_dim)
    
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
    
    def forward(self, inputs, anchor_idx, anchor_idx_tem, anchor_idx_dis):
        x = inputs.weight[1:,:]  # (N, D)
        anchor = inputs(anchor_idx)  # (r, D)
        anchor_tem = inputs(anchor_idx_tem)  # (r, D)
        anchor_dis = inputs(anchor_idx_dis)  # (r, D)
        
        anchor_adj_tra = self.cosine_matrix_div(x, anchor, type='tra')
        anchor_adj_tra = self.get_neighbor_hard_threshold(anchor_adj_tra, epsilon=self.tra_delta)  # (N, r)
        anchor_adj_tem, anchor_adj_dis = None, None
        if self.args.time_kl_reg > 0:
            anchor_adj_tem = self.cosine_matrix_div(x, anchor_tem, type='tem')
            anchor_adj_tem = self.get_neighbor_hard_threshold(anchor_adj_tem, epsilon=self.tem_delta)  # (N, r)
        if self.args.dis_kl_reg > 0:
            anchor_adj_dis = self.cosine_matrix_div(x, anchor_dis, type='dis')
            anchor_adj_dis = self.get_neighbor_hard_threshold(anchor_adj_dis, epsilon=self.dis_delta)  # (N, r)
        
        output_tra = self.gcn(x, anchor_adj_tra)
        mp_tra = torch.cat([inputs.weight[0, :].unsqueeze(dim=0), output_tra], dim=0)
        mp_tem, mp_dis = None, None
        if self.args.time_kl_reg > 0:
            output_tem = self.gcn(x, anchor_adj_tem)
            mp_tem = torch.cat([inputs.weight[0, :].unsqueeze(dim=0), output_tem], dim=0)
        if self.args.dis_kl_reg > 0:
            output_dis = self.gcn(x, anchor_adj_dis)
            mp_dis = torch.cat([inputs.weight[0, :].unsqueeze(dim=0), output_dis], dim=0)
        
        _, tra_indices = torch.topk(anchor_adj_tra, k=2, dim=-1)  # (N, 2)
        pos_neg_anchor_tra = torch.cat([anchor[tra_indices, :], output_tra.unsqueeze(1)], dim=1) # (N, 3, D)
        pos_neg_anchor_tem, pos_neg_anchor_dis = None, None
        if self.args.time_kl_reg > 0:
            _, tem_indices = torch.topk(anchor_adj_tem, k=2, dim=-1)
            pos_neg_anchor_tem = torch.cat([anchor_tem[tem_indices, :], output_tem.unsqueeze(1)], dim=1)  
        if self.args.dis_kl_reg > 0:
            _, dis_indices = torch.topk(anchor_adj_dis, k=2, dim=-1) 
            pos_neg_anchor_dis = torch.cat([anchor_dis[dis_indices, :], output_dis.unsqueeze(1)], dim=1)
        
        return mp_tra, mp_tem, mp_dis, anchor_adj_tra, anchor_adj_tem, anchor_adj_dis, pos_neg_anchor_tra, pos_neg_anchor_tem, pos_neg_anchor_dis