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

class FCD(nn.Module):

    def __init__(self, __C):
        super(FCD, self).__init__()
        self.linear_fcd = nn.Linear(__C.HIDDEN_SIZE * 2, __C.HIDDEN_SIZE)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, x_attend):

        fcd_x = self.sigmoid(self.linear_fcd(torch.cat((x, x_attend), dim=-1)))
        return fcd_x

class FMA(nn.Module):
    def __init__(self, __C):
        super(FMA, self).__init__()

        self.mhatt1 = MHAtt(__C)
        self.mhatt2 = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout3 = nn.Dropout(__C.DROPOUT_R)
        self.norm3 = LayerNorm(__C.HIDDEN_SIZE)

        self.fcd = FCD(__C)

    def forward(self, x, y, x_mask, y_mask):

        x_SA = self.mhatt1(x, x, x, x_mask)
        fcd_x_SA = self.fcd(x, x_SA)
        x = self.norm1(x + self.dropout1(fcd_x_SA * x_SA))

        x_GA = self.mhatt2(y, y, x, y_mask)
        fcd_x_GA = self.fcd(x, x_GA)
        x = self.norm2(x + self.dropout2(fcd_x_GA * x_GA))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x

class MCA(nn.Module):
    def __init__(self, __C):
        super(MCA, self).__init__()

        self.mhatt1 = MHAtt(__C)
        self.mhatt2 = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout3 = nn.Dropout(__C.DROPOUT_R)
        self.norm3 = LayerNorm(__C.HIDDEN_SIZE)

        self.fcd = FCD(__C)

    def forward(self, x, y, x_mask, y_mask):

        x = self.norm1(x + self.dropout1(self.mhatt1(x, x, x, x_mask)))
        x = self.norm2(x + self.dropout2(self.mhatt2(y, y, x, y_mask)))
        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x

class MCAN(nn.Module):
    def __init__(self, __C):
        super(MCAN, self).__init__()

        self.enc_list = nn.ModuleList([SA(__C) for _ in range(__C.LAYER)])
        # SGA : SA --> GA --> FFN
        self.dec_list = nn.ModuleList([MCA(__C) for _ in range(__C.LAYER)])
        print("__C.LAYER--:" + str(__C.LAYER))

    def forward(self, x, y, x_mask, y_mask):
        # Get hidden vector
        for enc in self.enc_list:
            x = enc(x, x_mask)

        for dec in self.dec_list:
            y = dec(y, x, y_mask, x_mask)

        return x, y