from core.model.net_utils import FC, MLP, LayerNorm, FFN

import torch.nn as nn
import torch.nn.functional as F
import torch, math
import copy



# ------------------------------------------------
# ---- MAoA Layers Cascaded by Encoder-Decoder ----
# ------------------------------------------------

"""
layer = 6
h = 8
hidden_size = 512
learing rate = min(2.5Te-5, 1e-4) T is the number of epoch
decay 0.2 after T=10 for every 2 epoch
13 epoch
batch size 64
0. L [2-8] D=6
"""

# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt_for_AoA(nn.Module):
    def  __init__(self, __C):
        super(MHAtt_for_AoA, self).__init__()
        self.__C = __C

        self.linear_v = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_k = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)

        self.linear_I_Q = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_I_Ved = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_G_Q = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_G_Ved = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)

        self.dropout = nn.Dropout(__C.DROPOUT_R)
        self.sigmoid = nn.Sigmoid()

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            self.__C.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            self.__C.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            self.__C.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.__C.HIDDEN_SIZE
        )

        I = self.linear_I_Q(q.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.__C.HIDDEN_SIZE
        )) + self.linear_I_Ved(atted)
        G = self.sigmoid(self.linear_G_Q(q.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.__C.HIDDEN_SIZE
        )) + self.linear_G_Ved(atted))

        output = I * G

        return output

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)

# ------------------------
# ---- Self Attention ----
# ------------------------

class SAoA(nn.Module):
    def __init__(self, __C):
        super(SAoA, self).__init__()

        self.mhatt = MHAtt_for_AoA(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, x, x_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt(x, x, x, x_mask)
        ))
        x = self.norm2(x + self.dropout2(
            self.ffn(x)
        ))

        return x


# -------------------------------
# ---- Self then Guided Attention ----
# -------------------------------

class SGAoA(nn.Module):
    def __init__(self, __C):
        super(SGAoA, self).__init__()

        self.mhatt1 = MHAtt_for_AoA(__C)
        self.mhatt2 = MHAtt_for_AoA(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout3 = nn.Dropout(__C.DROPOUT_R)
        self.norm3 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, x, y, x_mask, y_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt1(x, x, x, x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.mhatt2(y, y, x, y_mask)
        ))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x

class MCAoAN(nn.Module):
    def __init__(self, __C):
        super(MCAoAN, self).__init__()

        self.enc_list = nn.ModuleList([SAoA(__C) for _ in range(__C.LAYER)])
        # SGA : SA --> GA --> FFN
        self.dec_list = nn.ModuleList([SGAoA(__C) for _ in range(__C.LAYER)])
        print("__C.LAYER--:" + str(__C.LAYER))

    def forward(self, x, y, x_mask, y_mask):
        # Get hidden vector
        for enc in self.enc_list:
            x = enc(x, x_mask)

        for dec in self.dec_list:
            y = dec(y, x, y_mask, x_mask)

        return x, y