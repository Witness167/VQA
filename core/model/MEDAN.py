from core.model.net_utils import FC, MLP, LayerNorm, SA, MHAtt, FFN

import torch.nn as nn
import torch.nn.functional as F
import torch, math
import copy

"""
GA --> SA --> FFN
layer = 6
"""

class GSA(nn.Module):
    def __init__(self, __C):
        super(GSA, self).__init__()

        self.mhatt1 = MHAtt(__C)
        self.mhatt2 = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout3 = nn.Dropout(__C.DROPOUT_R)
        self.norm3 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, x, y, x_mask, y_mask):

        x = self.norm2(x + self.dropout2(
            self.mhatt2(y, y, x, y_mask)
        ))

        x = self.norm1(x + self.dropout1(
            self.mhatt1(x, x, x, x_mask)
        ))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x

# ------------------------------------------------
# ---- MEDAN Layers Cascaded by Encoder-Decoder ----
# ------------------------------------------------

class MEDAN(nn.Module):
    def __init__(self, __C):
        super(MEDAN, self).__init__()

        self.enc_list = nn.ModuleList([SA(__C) for _ in range(__C.LAYER)])
        # SGA : SA --> GA --> FFN
        self.dec_list = nn.ModuleList([GSA(__C) for _ in range(__C.LAYER)])
        print("__C.LAYER--:" + str(__C.LAYER))

    def forward(self, x, y, x_mask, y_mask):
        # Get hidden vector
        for enc in self.enc_list:
            x = enc(x, x_mask)

        for dec in self.dec_list:
            y = dec(y, x, y_mask, x_mask)

        return x, y

