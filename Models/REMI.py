import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import ast
import math

from Models.BasicModel import BasicModel, trans_to_cuda
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.Wq = nn.Linear(self.hidden_size, self.hidden_size)
        self.Wk = nn.Linear(self.hidden_size, self.hidden_size)
        self.Wv = nn.Linear(self.hidden_size, self.hidden_size)


    def forward(self, query, key, value):
        q = self.Wq(query)
        k = self.Wk(key)
        v = self.Wv(value)
        attn_scores = torch.bmm(q, k.transpose(1, 2)) / (
                key.size(-1) ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        context_vector = torch.bmm(attn_weights, v)

        return context_vector

class REMI(BasicModel):
    def __init__(self, opt, item_num, hidden_size, batch_size, interest_num=4, seq_len=50, add_pos=True, beta=0):
        super(REMI, self).__init__(item_num, hidden_size, batch_size, seq_len, beta)
        self.num_heads = interest_num
        self.interest_num = interest_num
        self.hard_readout = True
        self.add_pos = add_pos
        if self.add_pos:
            self.position_embedding = nn.Parameter(torch.Tensor(1, self.seq_len, self.hidden_size))
        self.linear1 = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size * 4, bias=False),
                nn.Tanh()
            )
        self.linear2 = nn.Linear(self.hidden_size * 4, self.num_heads, bias=False)
        self.text_hidden_size = 100
        self.attention = Attention(self.text_hidden_size)
        self.linear_text2seq = nn.Linear(self.text_hidden_size, hidden_size, bias=True)
        self.linear_text2seq2 = nn.Linear(self.text_hidden_size, hidden_size, bias=True)
        item_text_path = 'data/' + opt.dataset + '/{}_item_name_emb_100.csv'.format(opt.dataset)
        df = pd.read_csv(item_text_path)
        text_embeddings = df['Text_Embedding'].values
        text_pre_weight = [np.array(self.parse_embedding(temb)) for temb in text_embeddings]
        text_pre_weight = np.squeeze(np.array(text_pre_weight))

        self.item_text_dict = self.load_item_text_emb(item_text_path)
        self.item_text_embedding = nn.Embedding(self.item_num, self.text_hidden_size)
        self.item_text_embedding.weight.data.copy_(torch.from_numpy(text_pre_weight).float())
        self.aux_loss = nn.CrossEntropyLoss()

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def load_item_text_emb(self, file_path):
        df = pd.read_csv(file_path)
        item_text_emb_dict = {}
        for index, row in df.iterrows():
            key = row['item_id']
            value = row['Text_Embedding']
            item_text_emb_dict[key] = value
        return item_text_emb_dict

    def parse_embedding(self, embedding_str):
        embedding_values = ast.literal_eval(embedding_str)
        embedding_tensor = torch.tensor(embedding_values).float()
        return embedding_tensor
    def padding_embs(self, embs):
        max_length = max(emb.shape[0] for emb in embs)

        padded_embs = []
        masks = []

        for emb in embs:
            padding_length = max_length - emb.shape[0]

            padding_emb = torch.zeros(padding_length, emb.shape[1], dtype=emb.dtype)
            padded_emb = torch.cat([emb, padding_emb], dim=0)
            mask = torch.cat([torch.ones(emb.shape[0]), torch.zeros(padding_length)], dim=0).bool()
            padded_embs.append(padded_emb)

            masks.append(mask)

        padded_embs = trans_to_cuda(torch.stack(padded_embs, dim=0))
        masks = trans_to_cuda(torch.stack(masks, dim=0))

        return padded_embs, masks


    def forwardLogits(self, item_eb, mask):
        item_eb = item_eb * torch.reshape(mask, (-1, self.seq_len, 1))
        item_eb = torch.reshape(item_eb, (-1, self.seq_len, self.hidden_size))
        if self.add_pos:
            item_eb_add_pos = item_eb + self.position_embedding.repeat(item_eb.shape[0], 1, 1)
        else:
            item_eb_add_pos = item_eb
        item_hidden = self.linear1(item_eb_add_pos)
        item_att_w = self.linear2(item_hidden)
        item_att_w = torch.transpose(item_att_w, 2, 1).contiguous()
        atten_mask = torch.unsqueeze(mask, dim=1).repeat(1, self.num_heads, 1)
        paddings = torch.ones_like(atten_mask, dtype=torch.float) * (-2 ** 32 + 1)
        item_att_w = torch.where(torch.eq(atten_mask, 0), paddings, item_att_w)
        item_att_w = F.softmax(item_att_w, dim=-1)
        return item_att_w

    def calculate_text_score(self, user_text_eb):
        all_items = self.item_text_embedding.weight
        text_scores = torch.matmul(user_text_eb, all_items.transpose(1, 0))
        return text_scores

    def alignment_loss(self,readout, text_readout, item_eb, item_text_eb):
        def contrastive_loss(emb1, emb2, temperature=0.1):
            sim_matrix = F.cosine_similarity(emb1.unsqueeze(1), emb2.unsqueeze(0), dim=-1) / temperature
            pos_sim = torch.diag(sim_matrix)
            loss = -torch.log(torch.exp(pos_sim) / torch.exp(sim_matrix).sum(dim=1))
            loss = loss.mean()
            return loss

        def cosine_similarity_loss(emb1, emb2):
            cos_sim = F.cosine_similarity(emb1, emb2, dim=-1)
            loss = (1 - cos_sim).mean()
            return loss
        contrastive_loss_readout = contrastive_loss(readout, text_readout)
        contrastive_loss_item = contrastive_loss(item_eb, item_text_eb)
        cosine_loss_readout = cosine_similarity_loss(readout, text_readout)
        cosine_loss_item = cosine_similarity_loss(item_eb, item_text_eb)

        total_loss = 0.4*contrastive_loss_readout + 0.4*contrastive_loss_item + 0.1*cosine_loss_readout + 0.1*cosine_loss_item

        return total_loss

    def forward(self, item_list, label_list, mask, times, users, text_dict, cluster_dict, train=True):
        item_eb = self.embeddings(item_list)
        item_eb = item_eb * torch.reshape(mask, (-1, self.seq_len, 1))
        if train:
            label_eb = self.embeddings(label_list)
            label_text_eb = self.item_text_embedding(label_list)

        item_eb = torch.reshape(item_eb, (-1, self.seq_len, self.hidden_size))

        if self.add_pos:
            item_eb_add_pos = item_eb + self.position_embedding.repeat(item_eb.shape[0], 1, 1)
        else:
            item_eb_add_pos = item_eb

        item_hidden = self.linear1(item_eb_add_pos)
        item_att_w = self.linear2(item_hidden)
        item_att_w = torch.transpose(item_att_w, 2, 1).contiguous()

        atten_mask = torch.unsqueeze(mask, dim=1).repeat(1, self.num_heads, 1)
        paddings = torch.ones_like(atten_mask, dtype=torch.float) * (-2 ** 32 + 1)

        item_att_w = torch.where(torch.eq(atten_mask, 0), paddings, item_att_w)
        item_att_w = F.softmax(item_att_w, dim=-1)
        interest_emb = torch.matmul(item_att_w, item_eb)

        user_eb = interest_emb

        if not train:
            return user_eb, None

        typical_id_list = [cluster_dict[user] for user in users]
        text_emb_list = [self.parse_embedding(text_dict[id]) for id in typical_id_list]

        batch_text_embs, text_mask = self.padding_embs(text_emb_list)
        batch_text_embs = self.attention(batch_text_embs, batch_text_embs, batch_text_embs)
        batch_text_embs = batch_text_embs * text_mask.view(text_mask.shape[0], -1, 1).float()

        readout, selection = self.read_out(user_eb, label_eb)
        scores = None if self.is_sampler else self.calculate_score(readout)

        text_readout, text_selection = self.read_out_text(batch_text_embs, label_text_eb)
        text_scores = self.calculate_text_score(text_readout)

        text_readout = self.linear_text2seq(text_readout)
        label_text_eb = self.linear_text2seq2(label_text_eb)

        alignment_loss = self.alignment_loss(readout, text_readout, label_eb, label_text_eb)

        return user_eb, scores, text_scores, item_att_w, readout, selection, alignment_loss

    def calculate_atten_loss(self, attention):
        C_mean = torch.mean(attention, dim=2, keepdim=True)
        C_reg = (attention - C_mean)
        C_reg = torch.bmm(C_reg, C_reg.transpose(1, 2)) / self.hidden_size
        dr = torch.diagonal(C_reg, dim1=-2, dim2=-1)
        n2 = torch.norm(dr, dim=(1)) ** 2
        return n2.sum()