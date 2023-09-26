import numpy as np
import torch
from AGCN import AGCN_anchor
import sys
import torch.nn.functional as F

FLOAT_MIN = -sys.float_info.max

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs

class TimeAwareMultiHeadAttention(torch.nn.Module):
    def __init__(self, hidden_size, head_num, dropout_rate, dev):
        super(TimeAwareMultiHeadAttention, self).__init__()
        self.Q_w = torch.nn.Linear(hidden_size, hidden_size)
        self.K_w = torch.nn.Linear(hidden_size, hidden_size)
        self.V_w = torch.nn.Linear(hidden_size, hidden_size)

        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.softmax = torch.nn.Softmax(dim=-1)

        self.hidden_size = hidden_size
        self.head_num = head_num
        self.head_size = hidden_size // head_num
        self.dropout_rate = dropout_rate
        self.dev = dev

    def forward(self, queries, keys, time_mask, attn_mask, time_matrix_K, time_matrix_V, dis_matrix_K, dis_matrix_V, abs_pos_K, abs_pos_V):
        Q, K, V = self.Q_w(queries), self.K_w(keys), self.V_w(keys)

        Q_ = torch.cat(torch.split(Q, self.head_size, dim=2), dim=0)  # (B * h, N, D // h)
        K_ = torch.cat(torch.split(K, self.head_size, dim=2), dim=0)
        V_ = torch.cat(torch.split(V, self.head_size, dim=2), dim=0)

        time_matrix_K_ = torch.cat(torch.split(time_matrix_K, self.head_size, dim=3), dim=0)  # (B * h, N, N, D // h)
        time_matrix_V_ = torch.cat(torch.split(time_matrix_V, self.head_size, dim=3), dim=0)
        dis_matrix_K_ = torch.cat(torch.split(dis_matrix_K, self.head_size, dim=3), dim=0)
        dis_matrix_V_ = torch.cat(torch.split(dis_matrix_V, self.head_size, dim=3), dim=0)
        abs_pos_K_ = torch.cat(torch.split(abs_pos_K, self.head_size, dim=2), dim=0)
        abs_pos_V_ = torch.cat(torch.split(abs_pos_V, self.head_size, dim=2), dim=0)

        attn_weights = Q_.matmul(torch.transpose(K_, 1, 2))
        attn_weights += Q_.matmul(torch.transpose(abs_pos_K_, 1, 2))
        attn_weights += time_matrix_K_.matmul(Q_.unsqueeze(-1)).squeeze(-1)
        attn_weights += dis_matrix_K_.matmul((Q_.unsqueeze(-1))).squeeze(-1)

        attn_weights = attn_weights / (K_.shape[-1] ** 0.5)

        time_mask = time_mask.unsqueeze(-1).repeat(self.head_num, 1, 1)
        time_mask = time_mask.expand(-1, -1, attn_weights.shape[-1])
        attn_mask = attn_mask.unsqueeze(0).expand(attn_weights.shape[0], -1, -1)
        paddings = torch.ones(attn_weights.shape) *  (-2**32+1)
        paddings = paddings.to(self.dev)
        attn_weights = torch.where(time_mask, paddings, attn_weights)
        attn_weights = torch.where(attn_mask, paddings, attn_weights)

        attn_weights = self.softmax(attn_weights)
        attn_weights = self.dropout(attn_weights)

        outputs = attn_weights.matmul(V_)
        outputs += attn_weights.matmul(abs_pos_V_)
        outputs += attn_weights.unsqueeze(2).matmul(time_matrix_V_).reshape(outputs.shape).squeeze(2)
        outputs += attn_weights.unsqueeze(2).matmul(dis_matrix_V_).reshape(outputs.shape).squeeze(2)

        outputs = torch.cat(torch.split(outputs, Q.shape[0], dim=0), dim=2)

        return outputs


class MAGCN(torch.nn.Module):
    def __init__(self, user_num, item_num, time_num, args):
        super(MAGCN, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.args = args

        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.item_emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        if self.args.time_prompt:
            self.time_emb = torch.nn.Embedding(args.time_slot+1, args.hidden_units, padding_idx=0)
            self.time_emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.gcn = AGCN_anchor(args, input_dim=args.hidden_units, output_dim=args.hidden_units, layer=args.layer_num)

        self.abs_pos_K_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)
        self.abs_pos_V_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)
        self.time_matrix_K_emb = torch.nn.Embedding(args.time_span+1, args.hidden_units)
        self.time_matrix_V_emb = torch.nn.Embedding(args.time_span+1, args.hidden_units)
        self.dis_matrix_K_emb = torch.nn.Embedding(args.dis_span+1, args.hidden_units)
        self.dis_matrix_V_emb = torch.nn.Embedding(args.dis_span+1, args.hidden_units)

        self.abs_pos_K_emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.abs_pos_V_emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.time_matrix_K_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.time_matrix_V_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.dis_matrix_K_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.dis_matrix_V_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        # multiple graphs fusion
        self.project = torch.nn.Sequential(
            torch.nn.Linear(args.hidden_units, args.hidden_units),
            torch.nn.Linear(args.hidden_units, 1)
        )
    
        self.contra_intra = args.contra_intra
        self.contra_inter = args.contra_inter
        self.time_prompt = args.time_prompt
        self.dis_prompt = args.dis_prompt
        self.separate_loss = torch.nn.TripletMarginLoss(margin=1.0)

        for _ in range(args.num_blocks):  # 2
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)
            
            new_attn_layer = TimeAwareMultiHeadAttention(args.hidden_units, 
                                                         args.num_heads, 
                                                         args.dropout_rate, 
                                                         args.device)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)
    
    def seq2feats(self, user_ids, log_seqs, time_seqs, time_seqs_nxt, time_matrices, dis_matrices, item_embs):

        seqs = item_embs[torch.LongTensor(log_seqs).to(self.dev),:]  # (B, N, D)
        # seqs *= item_embs.shape[1] ** 0.5  #* why?
        
        seqs = self.item_emb_dropout(seqs)

        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])  # (B, N)
        positions = torch.LongTensor(positions).to(self.dev)
        abs_pos_K = self.abs_pos_K_emb(positions)
        abs_pos_V = self.abs_pos_V_emb(positions)
        abs_pos_K = self.abs_pos_K_emb_dropout(abs_pos_K)
        abs_pos_V = self.abs_pos_V_emb_dropout(abs_pos_V)

        time_matrices = torch.LongTensor(time_matrices).to(self.dev)
        time_matrix_K = self.time_matrix_K_emb(time_matrices)
        time_matrix_V = self.time_matrix_V_emb(time_matrices)
        time_matrix_K = self.time_matrix_K_dropout(time_matrix_K)
        time_matrix_V = self.time_matrix_V_dropout(time_matrix_V)

        dis_matrices = torch.LongTensor(dis_matrices).to(self.dev)
        dis_matrix_K = self.dis_matrix_K_emb(dis_matrices)
        dis_matrix_V = self.dis_matrix_V_emb(dis_matrices)
        dis_matrix_K = self.dis_matrix_K_dropout(dis_matrix_K)
        dis_matrix_V = self.dis_matrix_V_dropout(dis_matrix_V)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1)

        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            Q = self.attention_layernorms[i](seqs)
            mha_outputs = self.attention_layers[i](Q, seqs,  #* why do key==seqs rather than key==Q?
                                            timeline_mask, attention_mask,
                                            time_matrix_K, time_matrix_V,
                                            dis_matrix_K,dis_matrix_V,
                                            abs_pos_K, abs_pos_V)
            seqs = Q + mha_outputs
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        if self.args.time_prompt:
            seq_time = self.time_emb_dropout(self.time_emb(time_seqs_nxt.to(self.dev)))
            seqs += seq_time  # add next time embedding
        log_feats = self.last_layernorm(seqs)

        return log_feats

    def forward(self, user_ids, log_seqs, time_seqs, time_seqs_nxt, time_matrices, dis_matrices, pos_seqs, neg_seqs, anchor_idx, anchor_idx_tem, anchor_idx_dis, spatial_bias):

        anchor_idx = anchor_idx.to(self.dev)
        self.anchor_idx = anchor_idx
        anchor_idx_tem = anchor_idx_tem.to(self.dev)
        self.anchor_idx_tem = anchor_idx_tem
        anchor_idx_dis = anchor_idx_dis.to(self.dev)
        self.anchor_idx_dis = anchor_idx_dis
        item_embs, item_embs_tem, item_embs_dis, support, support_tem, support_dis, class_tra, class_tem, class_dis = self.gcn(self.item_emb, anchor_idx, anchor_idx_tem, anchor_idx_dis)
        
        if class_tem is None or class_dis is None:  # don't need contrastive learning, set 0
            contra_loss = torch.zeros(1).squeeze(0).to(self.dev)
        else:     
            # intra-graph contrastive loss
            pos_tra, neg_tra, anchor_tra = torch.split(class_tra, 1, dim=1)  # (N, 1, D)
            pos_tem, neg_tem, anchor_tem = torch.split(class_tem, 1, dim=1)  # (N, 1, D)
            pos_dis, neg_dis, anchor_dis = torch.split(class_dis, 1, dim=1)  # (N, 1, D)
            contra_loss_intra = self.separate_loss(pos_tra.squeeze(1), neg_tra.squeeze(1), anchor_tra.squeeze(1)) + \
            self.separate_loss(pos_tem.squeeze(1), neg_tem.squeeze(1), anchor_tem.squeeze(1)) + self.separate_loss(pos_dis.squeeze(1), neg_dis.squeeze(1), anchor_dis.squeeze(1))
            
            # inter-graph contrastive loss
            contra_loss_inter = self.separate_loss(pos_tem.squeeze(1), neg_tra.squeeze(1), anchor_tra.squeeze(1)) + \
                self.separate_loss(pos_dis.squeeze(1), neg_tra.squeeze(1), anchor_tra.squeeze(1)) + \
                self.separate_loss(pos_tra.squeeze(1), neg_tem.squeeze(1), anchor_tem.squeeze(1)) + \
                self.separate_loss(pos_dis.squeeze(1), neg_tem.squeeze(1), anchor_tem.squeeze(1)) + \
                self.separate_loss(pos_tra.squeeze(1), neg_dis.squeeze(1), anchor_dis.squeeze(1)) + \
                self.separate_loss(pos_tem.squeeze(1), neg_dis.squeeze(1), anchor_dis.squeeze(1))
            contra_loss = self.contra_intra * contra_loss_intra + self.contra_inter * contra_loss_inter
        
        # fusion
        if item_embs_tem is not None and item_embs_dis is not None:
            item_embs_score, item_embs_tem_score, item_embs_dis_score = self.project(item_embs), self.project(item_embs_tem), self.project(item_embs_dis)  # (N, 1)
            adaptive_score = torch.softmax(torch.exp(torch.cat([item_embs_score, item_embs_tem_score, item_embs_dis_score], dim=-1)), dim=-1)
            item_embs = torch.sum(adaptive_score.unsqueeze(-1) * torch.stack([item_embs, item_embs_tem, item_embs_dis], dim=1), dim=1)
        elif item_embs_tem is not None:
            item_embs_score, item_embs_tem_score = self.project(item_embs), self.project(item_embs_tem)  # (N, 1)
            adaptive_score = torch.softmax(torch.exp(torch.cat([item_embs_score, item_embs_tem_score], dim=-1)), dim=-1)
            item_embs = torch.sum(adaptive_score.unsqueeze(-1) * torch.stack([item_embs, item_embs_tem], dim=1), dim=1)
        elif item_embs_dis is not None:
            item_embs_score, item_embs_dis_score = self.project(item_embs), self.project(item_embs_dis)  # (N, 1)
            adaptive_score = torch.softmax(torch.exp(torch.cat([item_embs_score, item_embs_dis_score], dim=-1)), dim=-1)
            item_embs = torch.sum(adaptive_score.unsqueeze(-1) * torch.stack([item_embs, item_embs_dis], dim=1), dim=1)
            
        log_feats = self.seq2feats(user_ids, log_seqs, time_seqs, time_seqs_nxt, time_matrices, dis_matrices, item_embs)  # (B, N, D)
        pos_embs = item_embs[torch.LongTensor(pos_seqs).to(self.dev),:]  # label
        neg_embs = item_embs[torch.LongTensor(neg_seqs).to(self.dev),:]  # useless

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        fin_logits = log_feats.matmul(item_embs.transpose(0,1))
        #* spatial_preference
        if self.args.dis_prompt:
            bias = spatial_bias[torch.LongTensor(log_seqs), :].to(self.dev)  # (B, N, N_all)
            fin_logits = fin_logits + bias
        fin_logits = fin_logits.reshape(-1, fin_logits.shape[-1])  # (B*N, N_all)
        
        return pos_logits, neg_logits, fin_logits, support, support_tem, support_dis, contra_loss

    def predict(self, user_ids, log_seqs, time_seq, time_seq_nxt, time_matrices, dis_matrices, item_indices, spatial_bias):
        item_embs, item_embs_tem, item_embs_dis, support, support_tem, support_dis, class_tra, class_tem, class_dis = self.gcn(self.item_emb, self.anchor_idx, self.anchor_idx_tem, self.anchor_idx_dis)
        #* fusion
        # fusion
        if item_embs_tem is not None and item_embs_dis is not None:
            item_embs_score, item_embs_tem_score, item_embs_dis_score = self.project(item_embs), self.project(item_embs_tem), self.project(item_embs_dis)  # (N, 1)
            adaptive_score = torch.softmax(torch.exp(torch.cat([item_embs_score, item_embs_tem_score, item_embs_dis_score], dim=-1)), dim=-1)
            item_embs = torch.sum(adaptive_score.unsqueeze(-1) * torch.stack([item_embs, item_embs_tem, item_embs_dis], dim=1), dim=1)
        elif item_embs_tem is not None:
            item_embs_score, item_embs_tem_score = self.project(item_embs), self.project(item_embs_tem)  # (N, 1)
            adaptive_score = torch.softmax(torch.exp(torch.cat([item_embs_score, item_embs_tem_score], dim=-1)), dim=-1)
            item_embs = torch.sum(adaptive_score.unsqueeze(-1) * torch.stack([item_embs, item_embs_tem], dim=1), dim=1)
        elif item_embs_dis is not None:
            item_embs_score, item_embs_dis_score = self.project(item_embs), self.project(item_embs_dis)  # (N, 1)
            adaptive_score = torch.softmax(torch.exp(torch.cat([item_embs_score, item_embs_dis_score], dim=-1)), dim=-1)
            item_embs = torch.sum(adaptive_score.unsqueeze(-1) * torch.stack([item_embs, item_embs_dis], dim=1), dim=1)
        
        log_feats = self.seq2feats(user_ids, log_seqs, time_seq, time_seq_nxt, time_matrices, dis_matrices, item_embs)
        final_feat = log_feats[:, -1, :]
        logits = final_feat.matmul(item_embs.transpose(0,1))
        #* spatial_preference
        if self.args.dis_prompt:
            bias = spatial_bias[torch.LongTensor(log_seqs), :].to(self.dev)  # (B, N, N_all)
            logits = logits + bias[:, -1, :]
        
        return logits, item_indices