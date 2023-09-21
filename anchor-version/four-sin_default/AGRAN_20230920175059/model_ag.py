import numpy as np
import torch
from AGCN import AGCN_anchor
import sys
import torch.nn.functional as F

FLOAT_MIN = -sys.float_info.max

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

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
    
class FilterLayer(torch.nn.Module):
    def __init__(self, max_len, hidden_units, dropout_rate):
        super(FilterLayer, self).__init__()
        self.complex_weight = torch.nn.Parameter(torch.randn(1, max_len//2 + 1, hidden_units, 2, dtype=torch.float32) * 0.02)
        self.out_dropout = torch.nn.Dropout(dropout_rate)
        self.LayerNorm = torch.nn.LayerNorm(hidden_units, eps=1e-18)


    def forward(self, input_tensor):
        # [batch, seq_len, hidden]
        batch, seq_len, hidden = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        sequence_emb_fft = torch.fft.irfft(x, n=seq_len, dim=1, norm='ortho')
        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


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
        # self.user_K = torch.nn.Linear(hidden_size, hidden_size)
        # self.user_V = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, queries, keys, time_mask, attn_mask, time_matrix_K, time_matrix_V, dis_matrix_K, dis_matrix_V, abs_pos_K, abs_pos_V, bias=None, user_bias=None):
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
        # user_bias_K_ = torch.cat(torch.split(self.user_K(user_bias), self.head_size, dim=2), dim=0)  # (B * h, N, D // h)
        # user_bias_V_ = torch.cat(torch.split(self.user_V(user_bias), self.head_size, dim=2), dim=0)  # (B * h, N, D // h)
        # user_bias_K_ = torch.cat(torch.split(user_bias_K, self.head_size, dim=2), dim=0)  # (B * h, N, D // h)
        # user_bias_V_ = torch.cat(torch.split(user_bias_V, self.head_size, dim=2), dim=0)  # (B * h, N, D // h)

        attn_weights = Q_.matmul(torch.transpose(K_, 1, 2))
        attn_weights += Q_.matmul(torch.transpose(abs_pos_K_, 1, 2))
        attn_weights += time_matrix_K_.matmul(Q_.unsqueeze(-1)).squeeze(-1)
        attn_weights += dis_matrix_K_.matmul((Q_.unsqueeze(-1))).squeeze(-1)
        # attn_weights += user_bias
        # attn_weights += Q_.matmul(torch.transpose(user_bias_K_, 1, 2))
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
        # outputs += attn_weights.matmul(user_bias_V_)
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
        #* multiple graphs fusion
        self.project = torch.nn.Sequential(
            torch.nn.Linear(args.hidden_units, args.hidden_units),
            torch.nn.Linear(args.hidden_units, 1)
        )
        self.recon_tra = 1.0
        self.recon_tem = 1.0
        self.recon_dis = 1.0
        
        # self.filter_layers = torch.nn.ModuleList()
        # self.classifier = torch.nn.Sequential(
        #     # torch.nn.Linear(args.hidden_units, args.hidden_units),
        #     torch.nn.Linear(args.hidden_units, args.anchor_num)
        # )
        # self.degree_pos_K_emb = torch.nn.Embedding(1, args.hidden_units)
        # self.degree_pos_V_emb = torch.nn.Embedding(1, args.hidden_units)
        # self.degree_pos_K_emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        # self.degree_pos_V_emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        for _ in range(args.num_blocks):  # 2
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)
            #* filter layers
            # new_filter_layer = FilterLayer(args.maxlen, args.hidden_units, args.dropout_rate)
            # self.filter_layers.append(new_filter_layer)
            
            new_attn_layer = TimeAwareMultiHeadAttention(args.hidden_units, 
                                                         args.num_heads, 
                                                         args.dropout_rate, 
                                                         args.device)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)
    
    def seq2feats(self, user_ids, log_seqs, time_matrices, dis_matrices, item_embs, interaction_matrix, spatial_bias, user_bias):

        seqs = item_embs[torch.LongTensor(log_seqs).to(self.dev),:]  # (B, N, D)
        # seqs *= item_embs.shape[1] ** 0.5  #* why?
        seqs = self.item_emb_dropout(seqs)
        bias = spatial_bias[torch.LongTensor(log_seqs), :].to(self.dev)  # (B, N, N_all)
        # user_bias = user_bias[torch.LongTensor(user_ids), :].to(self.dev)  # (B, N_all)
        # user_bias = user_bias[torch.arange(log_seqs.size(0)).unsqueeze(1), log_seqs].unsqueeze(-1)  # (B, N, 1)
        # user_bias_K = user_bias * self.degree_pos_K_emb.weight.unsqueeze(0)  # (B, N, D)
        # user_bias_V = user_bias * self.degree_pos_V_emb.weight.unsqueeze(0)  # (B, N, D)
        # user_bias_K = self.degree_pos_K_emb_dropout(user_bias_K)
        # user_bias_V = self.degree_pos_V_emb_dropout(user_bias_V)

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
            # Q = self.filter_layers[i](seqs)
            mha_outputs = self.attention_layers[i](Q, seqs,  #* why do key==seqs rather than key==Q?
                                            timeline_mask, attention_mask,
                                            time_matrix_K, time_matrix_V,
                                            dis_matrix_K,dis_matrix_V,
                                            abs_pos_K, abs_pos_V, bias, user_bias)
            seqs = Q + mha_outputs
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)

        return log_feats

    def forward(self, user_ids, log_seqs, time_matrices, dis_matrices, pos_seqs, neg_seqs, anchor_idx, anchor_idx_tem, anchor_idx_dis, time_adj_matrix=None, dis_adj_matrix=None, spatial_bias=None, interaction_matrix=None):

        anchor_idx = anchor_idx.to(self.dev)
        self.anchor_idx = anchor_idx
        anchor_idx_tem = anchor_idx_tem.to(self.dev)
        self.anchor_idx_tem = anchor_idx_tem
        anchor_idx_dis = anchor_idx_dis.to(self.dev)
        self.anchor_idx_dis = anchor_idx_dis
        item_embs, item_embs_tem, item_embs_dis, support, support_tem, support_dis, class_tra, class_tem, class_dis = self.gcn(self.item_emb, anchor_idx, anchor_idx_tem, anchor_idx_dis, time_adj_matrix, dis_adj_matrix)
        
        #* reconstruction loss
        # recon_loss_tra = calculate_reconstruct_loss(support, item_embs, anchor_idx)
        # recon_loss_tem = calculate_reconstruct_loss(support_tem, item_embs_tem, anchor_idx_tem)
        # recon_loss_dis = calculate_reconstruct_loss(support_dis, item_embs_dis, anchor_idx_dis)
        # recon_loss = self.recon_tra * recon_loss_tra + self.recon_tem * recon_loss_tem + self.recon_dis * recon_loss_dis
        
        #* classification loss
        # tra_class_loss = F.cross_entropy(self.classifier(item_embs[1:,]), class_tra)
        # tem_class_loss = F.cross_entropy(self.classifier(item_embs_tem[1:,]), class_tem)
        # dis_class_loss = F.cross_entropy(self.classifier(item_embs_dis[1:,]), class_dis)
        # recon_loss = self.recon_tra * tra_class_loss + self.recon_tem * tem_class_loss + self.recon_dis * dis_class_loss
        recon_loss = torch.zeros(1).to(self.dev)
        
        #* contrastive learning
        non_anchor_idx = set(range(1, item_embs.shape[0])).difference(set(anchor_idx.cpu().numpy().tolist()).union(set(anchor_idx_tem.cpu().numpy().tolist())))
        random_non_anchor_idx = np.random.choice(np.array(list(non_anchor_idx)), min(len(anchor_idx), len(non_anchor_idx)), replace=False)
        random_non_anchor_idx = torch.from_numpy(random_non_anchor_idx).to(self.dev)  # (N', )
        
        non_item_embs, non_tem_item_embs = item_embs[random_non_anchor_idx, :], item_embs_tem[random_non_anchor_idx, :]  # (N', D)
        # sim_score = torch.exp(F.cosine_similarity(non_item_embs.unsqueeze(1), non_tem_item_embs.unsqueeze(0), dim=-1) / self.temp)  # (N', N')
        # diag = torch.eye(non_item_embs.shape[0], dtype=torch.float32).to(self.dev)  # (N', N')
        # pos_score = torch.sum(sim_score * diag, dim=-1)  # (N', )
        # neg_score = torch.sum(sim_score, dim=-1)
        # ratio = (pos_score + 1e-12) / neg_score
        # contra_loss = torch.mean(-torch.log(ratio))
        
        sim_score = non_item_embs.T @ non_tem_item_embs  # (D, D)
        # sim_score1 = F.cosine_similarity(non_item_embs.transpose(0, 1).unsqueeze(1), tem_item_embs.transpose(0, 1).unsqueeze(0), dim=-1)  # (D, D) 
        # sim_score2 = F.cosine_similarity(non_item_embs.transpose(0, 1).unsqueeze(1), dis_item_embs.transpose(0, 1).unsqueeze(0), dim=-1)
        # contra_loss = torch.sum((sim_score1 - diag) ** 2) + torch.sum((sim_score2 - diag) ** 2)
        on_diag_intra = torch.diagonal(sim_score).add_(-1).pow_(2).sum()
        off_diag_intra = off_diagonal(sim_score).pow_(2).sum()
        contra_loss = on_diag_intra + 1e-3 * off_diag_intra
        
        #* fusion
        item_embs_score, item_embs_tem_score, item_embs_dis_score = self.project(item_embs), self.project(item_embs_tem), self.project(item_embs_dis)  # (N, 1)
        adaptive_score = torch.softmax(torch.exp(torch.cat([item_embs_score, item_embs_tem_score, item_embs_dis_score], dim=-1)), dim=-1)
        item_embs = torch.sum(adaptive_score.unsqueeze(-1) * torch.stack([item_embs, item_embs_tem, item_embs_dis], dim=1), dim=1)
        
        #* user preference
        # user_preference = interaction_matrix.matmul(item_embs)  # (M, D) self.item_emb.weight
        # user_bias = user_preference[torch.LongTensor(user_ids), :].unsqueeze(1).to(self.dev)  # (B, 1, D)
        # user_bias = torch.exp(-torch.norm(user_bias - item_embs.unsqueeze(0), p=2, dim=-1)).unsqueeze(1)  # (B, 1, N_all)
        user_bias = interaction_matrix
        
        log_feats = self.seq2feats(user_ids, log_seqs, time_matrices, dis_matrices, item_embs, interaction_matrix, spatial_bias, user_bias)  # (B, N, D)
        pos_embs = item_embs[torch.LongTensor(pos_seqs).to(self.dev),:]  # label
        neg_embs = item_embs[torch.LongTensor(neg_seqs).to(self.dev),:]  # useless

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        fin_logits = log_feats.matmul(item_embs.transpose(0,1))
        #* spatial_preference
        bias = spatial_bias[torch.LongTensor(log_seqs), :].to(self.dev)  # (B, N, N_all)
        fin_logits = fin_logits + bias
        # fin_logits = fin_logits + user_bias  #* user bias
        fin_logits = fin_logits.reshape(-1,fin_logits.shape[-1])  # (B*N, N_all)
        
        return self.item_emb.weight, pos_logits, neg_logits, fin_logits, support, support_tem, support_dis, contra_loss, recon_loss

    def predict(self, user_ids, log_seqs, time_matrices, dis_matrices, item_indices, time_adj_matrix=None, dis_adj_matrix=None, spatial_bias=None, interaction_matrix=None):
        item_embs, item_embs_tem, item_embs_dis, support, support_tem, support_dis, class_tra, class_tem, class_dis = self.gcn(self.item_emb, self.anchor_idx, self.anchor_idx_tem, self.anchor_idx_dis, time_adj_matrix, dis_adj_matrix)
        # #* fusion
        # item_embs_score, item_embs_tem_score = self.project(item_embs), self.project(item_embs_tem)  # (N, 1)
        # adaptive_score = torch.softmax(torch.exp(torch.cat([item_embs_score, item_embs_tem_score], dim=-1)), dim=-1)
        # item_embs = torch.sum(adaptive_score.unsqueeze(-1) * torch.stack([item_embs, item_embs_tem], dim=1), dim=1)
        #* fusion
        item_embs_score, item_embs_tem_score, item_embs_dis_score = self.project(item_embs), self.project(item_embs_tem), self.project(item_embs_dis)  # (N, 1)
        adaptive_score = torch.softmax(torch.exp(torch.cat([item_embs_score, item_embs_tem_score, item_embs_dis_score], dim=-1)), dim=-1)
        item_embs = torch.sum(adaptive_score.unsqueeze(-1) * torch.stack([item_embs, item_embs_tem, item_embs_dis], dim=1), dim=1)
        # #* user preference
        # user_preference = interaction_matrix.matmul(item_embs)  # (M, D)self.item_emb.weight
        # user_bias = user_preference[torch.LongTensor(user_ids), :].unsqueeze(1).to(self.dev)  # (B, 1, D)
        # user_bias = torch.exp(-torch.norm(user_bias - item_embs.unsqueeze(0), p=2, dim=-1))  # (B, N_all)
        user_bias = interaction_matrix
        
        log_feats = self.seq2feats(user_ids, log_seqs, time_matrices, dis_matrices, item_embs, interaction_matrix, spatial_bias, user_bias)
        final_feat = log_feats[:, -1, :]
        logits = final_feat.matmul(item_embs.transpose(0,1))
        #* spatial_preference
        bias = spatial_bias[torch.LongTensor(log_seqs), :].to(self.dev)  # (B, N, N_all)
        logits = logits + bias[:, -1, :]
        # logits = logits + user_bias
        
        return logits, item_indices

def calculate_reconstruct_loss(support, emb, anchor_idx):
    x = emb[1:, ]
    anchor = emb[anchor_idx]
    node_norm = x.div(torch.norm(x, p=2, dim=-1, keepdim=True))
    anchor_norm = anchor.div(torch.norm(anchor, p=2, dim=-1, keepdim=True))
    cos_adj = torch.mm(node_norm, anchor_norm.transpose(-1, -2))  # (N, r)
    return F.mse_loss(cos_adj, support)