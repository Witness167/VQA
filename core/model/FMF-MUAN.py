import torch.nn as nn
from .net_utils import FFN, LayerNorm
import torch
import math
import torch.nn.functional as F

"""
HIDDEN_SIZE, d= 768
h = 8
d/h = 96
d_g = 96
# python run.py --RUN="train" --GPU="2" --BACKBONE="MUAN" --LAYER=8 --VERSION="MUAN-8-test0" --NW=16 --HIDDEN_SIZE=768 --LR_BASE=0.00019 --SPLIT="train"
"""

class GatedSA(nn.Module):
    def __init__(self, __C):
        super(GatedSA, self).__init__()

        self.__C = __C
        self.linear_v = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_k = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_g_k = nn.Linear(__C.HIDDEN_SIZE, __C.DG)
        self.linear_g_q = nn.Linear(__C.HIDDEN_SIZE, __C.DG)
        self.linear_g =  nn.Linear(__C.DG, 2)
        self.sigmoid_mask = nn.Sigmoid()
        self.dropout_attention_map = nn.Dropout(__C.DROPOUT_R)

    def forward(self, v, k, q, mask):
        batch_size = q.size(0)

        v = self.linear_v(v); k = self.linear_k(k); q = self.linear_q(q)
        mask_qk = self.sigmoid_mask(self.linear_g(self.linear_g_q(q) * self.linear_g_k(k)))
        mask_k = mask_qk[:,:,0]
        mask_q = mask_qk[:,:,-1]
        k_masked = mask_k.unsqueeze(-1).expand(-1, -1, self.__C.HIDDEN_SIZE) * k
        q_masked = mask_q.unsqueeze(-1).expand(-1, -1, self.__C.HIDDEN_SIZE) * q
        v = v.view(batch_size, -1, self.__C.MULTI_HEAD, self.__C.HIDDEN_SIZE_HEAD).transpose(1, 2)
        k_masked = k_masked.view(batch_size, -1, self.__C.MULTI_HEAD, self.__C.HIDDEN_SIZE_HEAD).transpose(1, 2)
        q_masked = q_masked.view(batch_size, -1, self.__C.MULTI_HEAD, self.__C.HIDDEN_SIZE_HEAD).transpose(1, 2)

        v_attended = self.att(v, k_masked, q_masked, mask)
        v_attended = v_attended.transpose(1, 2).contiguous().view(batch_size, -1, self.__C.HIDDEN_SIZE)
        return v_attended

    def att(self, v, k, q, mask):
        d_k = k.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        attention_map = F.softmax(scores, dim=-1)
        attention_map = self.dropout_attention_map(attention_map)
        v_attended = torch.matmul(attention_map, v)
        return v_attended

class UA(nn.Module):
    def __init__(self, __C):
        super(UA, self).__init__()

        self.gatedSA = GatedSA(__C)
        self.ffn = FFN(__C)
        self.norm_0 = LayerNorm(__C.HIDDEN_SIZE)
        self.dropout_0 = nn.Dropout(__C.DROPOUT_R)
        self.norm_1 = LayerNorm(__C.HIDDEN_SIZE)
        self.dropout_1 = nn.Dropout(__C.DROPOUT_R)

    def forward(self, z, z_mask):

        z = self.norm_0(z + self.dropout_0(self.gatedSA(z, z, z, z_mask)))
        z = self.norm_1(z + self.dropout_1(self.ffn(z)))

        return z

class FU(nn.Module):

    def __init__(self, __C):
        super(FU, self).__init__()

        self.UAs = nn.ModuleList([UA(__C) for _ in range(__C.LAYER)])
        print("{ __C.LAYER %-17s }" % __C.LAYER)

    def forward(self, x, y, x_mask, y_mask):
        z = torch.cat([x, y], dim=1)

        z_mask = (torch.sum(
            torch.abs(z),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)

        for UA in self.UAs:
            z = UA(z, z_mask)

        return z