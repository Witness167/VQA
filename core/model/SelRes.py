from core.model.net_utils import LayerNorm, SA, MHAtt, FFN

import torch.nn as nn
import torch.nn.functional as F
import torch, math
import copy



# ------------------------------------------------
# ---- MAC Layers Cascaded by Encoder-Decoder ----
# ------------------------------------------------

"""
layer = 6
image feature 2048
question feature 512 
fused feature 1024
latent d 512 multi-head 8   d/h=64
beta1 2 = 0.9 0.98
learning rate 0.001 fro first 2 epoch
decay after 10 epochs 0.002
decay rate 0.25 each 2 epochs
train 13 epochs 
batch size 64
"""

class MHAtt_for_SelRes(nn.Module):
    def  __init__(self, __C):
        super(MHAtt_for_SelRes, self).__init__()
        self.__C = __C

        self.linear_Sel_threshold = nn.Linear(__C.HIDDEN_SIZE, 1)
        self.linear_v = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_k = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_merge = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)

        self.dropout = nn.Dropout(__C.DROPOUT_R)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            self.__C.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        k = self.linear_k(k)
        q = self.linear_q(q)
        threshold = self.linear_Sel_threshold(q * k) #torch.Size([64, 100, 1])

        k = k.view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            self.__C.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        q = q.view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            self.__C.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        att_map = self.att(k, q, mask) # {bs, 8, n, n}#torch.Size([64, 8, 100, 100])
        a = ((torch.sum(torch.sum(att_map, dim=-2), dim=-2).unsqueeze(-1) - threshold) > 0)
        atted = torch.matmul(att_map, v)

        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.__C.HIDDEN_SIZE
        )
        atted = atted * a

        atted = self.linear_merge(atted)

        return atted

    def att(self, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        #att_map = self.dropout(att_map)

        return att_map

class SelResGA(nn.Module):
    def __init__(self, __C):
        super(SelResGA, self).__init__()

        self.mhatt1 = MHAtt_for_SelRes(__C)
        self.mhatt2 = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout3 = nn.Dropout(__C.DROPOUT_R)
        self.norm3 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, x, y, x_mask, y_mask):

        x = self.norm1(x + self.dropout1(self.mhatt1(x, x, x, x_mask)))

        x = self.norm2(x + self.dropout2(self.mhatt2(y, y, x, y_mask)))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x

class SelRes(nn.Module):
    def __init__(self, __C):
        super(SelRes, self).__init__()

        self.enc_list = nn.ModuleList([SA(__C) for _ in range(__C.LAYER)])
        # SGA : SA --> GA --> FFN
        self.dec_list = nn.ModuleList([SelResGA(__C) for _ in range(__C.LAYER)])
        print("__C.LAYER--:" + str(__C.LAYER))

    def forward(self, x, y, x_mask, y_mask):
        # Get hidden vector
        for enc in self.enc_list:
            x = enc(x, x_mask)

        for dec in self.dec_list:
            y = dec(y, x, y_mask, x_mask)

        return x, y