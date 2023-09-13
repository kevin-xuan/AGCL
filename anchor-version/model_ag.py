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
        # self.fusion_k = torch.nn.Linear(2 * hidden_size, hidden_size)
        # self.fusion_v = torch.nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, queries, keys, time_mask, attn_mask, time_matrix_K, time_matrix_V, dis_matrix_K, dis_matrix_V, abs_pos_K, abs_pos_V):
        Q, K, V = self.Q_w(queries), self.K_w(keys), self.V_w(keys)

        Q_ = torch.cat(torch.split(Q, self.head_size, dim=2), dim=0)
        K_ = torch.cat(torch.split(K, self.head_size, dim=2), dim=0)
        V_ = torch.cat(torch.split(V, self.head_size, dim=2), dim=0)

        time_matrix_K_ = torch.cat(torch.split(time_matrix_K, self.head_size, dim=3), dim=0)
        time_matrix_V_ = torch.cat(torch.split(time_matrix_V, self.head_size, dim=3), dim=0)
        dis_matrix_K_ = torch.cat(torch.split(dis_matrix_K, self.head_size, dim=3), dim=0)
        dis_matrix_V_ = torch.cat(torch.split(dis_matrix_V, self.head_size, dim=3), dim=0)
        abs_pos_K_ = torch.cat(torch.split(abs_pos_K, self.head_size, dim=2), dim=0)
        abs_pos_V_ = torch.cat(torch.split(abs_pos_V, self.head_size, dim=2), dim=0)

        attn_weights = Q_.matmul(torch.transpose(K_, 1, 2))
        attn_weights += Q_.matmul(torch.transpose(abs_pos_K_, 1, 2))
        attn_weights += time_matrix_K_.matmul(Q_.unsqueeze(-1)).squeeze(-1)
        attn_weights += dis_matrix_K_.matmul((Q_.unsqueeze(-1))).squeeze(-1)
        # attn_weights += self.fusion_k(torch.cat([time_matrix_K_, dis_matrix_K_], dim=-1)).matmul(Q_.unsqueeze(-1)).squeeze(-1)

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
        # outputs += attn_weights.unsqueeze(2).matmul(self.fusion_v(torch.cat([time_matrix_V_, dis_matrix_V_], dim=-1))).reshape(outputs.shape).squeeze(2)

        outputs = torch.cat(torch.split(outputs, Q.shape[0], dim=0), dim=2)

        return outputs


class AGRAN_anchor(torch.nn.Module):
    def __init__(self, user_num, item_num, time_num, args):
        super(AGRAN_anchor, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device

        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.item_emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.gcn = AGCN_anchor(input_dim=args.hidden_units, output_dim=args.hidden_units, layer=args.layer_num)

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
        #* contrastive learning
        self.project = torch.nn.Sequential(
            torch.nn.Linear(args.hidden_units, args.hidden_units),
            torch.nn.Linear(args.hidden_units, 1)
        )

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

    def seq2feats(self, user_ids, log_seqs, time_matrices, dis_matrices, item_embs):

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

        log_feats = self.last_layernorm(seqs)

        return log_feats

    def forward(self, user_ids, log_seqs, time_matrices, dis_matrices, pos_seqs, neg_seqs, anchor_idx, time_adj_matrix=None, dis_adj_matrix=None):

        anchor_idx = anchor_idx.to(self.dev)
        self.anchor_idx = anchor_idx
        item_embs, support, mp_tem, mp_dis = self.gcn(self.item_emb, anchor_idx, time_adj_matrix, dis_adj_matrix)
        #* contrastive learning
        non_anchor_idx = set(range(1, item_embs.shape[0])).difference(set(anchor_idx.cpu().numpy().tolist()))
        non_anchor_idx = torch.from_numpy(np.array(list(non_anchor_idx))).to(self.dev)
        non_item_embs, tem_item_embs, dis_item_embs = item_embs[non_anchor_idx, :], mp_tem[non_anchor_idx, :], mp_dis[non_anchor_idx, :]  # (N-r, D)
        diag = torch.eye(non_item_embs.shape[1], dtype=torch.float32).to(self.dev)  # (D, D)
        
        sim_score1 = F.cosine_similarity(non_item_embs.transpose(0, 1).unsqueeze(1), tem_item_embs.transpose(0, 1).unsqueeze(0), dim=-1)  # (D, D) 
        sim_score2 = F.cosine_similarity(non_item_embs.transpose(0, 1).unsqueeze(1), dis_item_embs.transpose(0, 1).unsqueeze(0), dim=-1)
        contra_loss = torch.sum((sim_score1 - diag) ** 2) + torch.sum((sim_score2 - diag) ** 2)
        #* fusion
        item_embs_score, mp_tem_score, mp_dis_score = self.project(item_embs), self.project(mp_tem), self.project(mp_dis)  # (N, 1)
        adaptive_score = torch.softmax(torch.exp(torch.cat([item_embs_score, mp_tem_score, mp_dis_score], dim=-1)), dim=-1)
        item_embs = torch.sum(adaptive_score.unsqueeze(-1) * torch.stack([item_embs, mp_tem, mp_dis], dim=1), dim=1)
        
        
        log_feats = self.seq2feats(user_ids, log_seqs, time_matrices, dis_matrices, item_embs)  # (B, N, D)

        pos_embs = item_embs[torch.LongTensor(pos_seqs).to(self.dev),:]  # label
        neg_embs = item_embs[torch.LongTensor(neg_seqs).to(self.dev),:]  # useless

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        fin_logits = log_feats.matmul(item_embs.transpose(0,1))
        fin_logits = fin_logits.reshape(-1,fin_logits.shape[-1])  # (B*N, N_all)
        
        return self.item_emb.weight, pos_logits, neg_logits, fin_logits, support, contra_loss

    def predict(self, user_ids, log_seqs, time_matrices, dis_matrices, item_indices, time_adj_matrix=None, dis_adj_matrix=None):
        item_embs, support, mp_tem, mp_dis = self.gcn(self.item_emb, self.anchor_idx, time_adj_matrix, dis_adj_matrix)
        #* fusion
        item_embs_score, mp_tem_score, mp_dis_score = self.project(item_embs), self.project(mp_tem), self.project(mp_dis)  # (N, 1)
        adaptive_score = torch.softmax(torch.exp(torch.cat([item_embs_score, mp_tem_score, mp_dis_score], dim=-1)), dim=-1)
        item_embs = torch.sum(adaptive_score.unsqueeze(-1) * torch.stack([item_embs, mp_tem, mp_dis], dim=1), dim=1)
        
        log_feats = self.seq2feats(user_ids, log_seqs, time_matrices, dis_matrices, item_embs)
        final_feat = log_feats[:, -1, :]
        logits = final_feat.matmul(item_embs.transpose(0,1))

        return logits, item_indices
