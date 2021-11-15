from core.model.net_utils import FC, MLP, LayerNorm, SA, SGA

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
best epoch is chosen in the validation set

"""

class DCAN(nn.Module):
    def __init__(self, __C):
        super(DCAN, self).__init__()

        self.enc_list_0 = nn.ModuleList([SA(__C) for _ in range(__C.LAYER)])
        self.enc_list_1 = nn.ModuleList([SA(__C) for _ in range(__C.LAYER)])
        # SGA : SA --> GA --> FFN
        self.dec_list = nn.ModuleList([SGA(__C) for _ in range(__C.LAYER)])
        print("__C.LAYER--:" + str(__C.LAYER))

    def forward(self, x, y, x_mask, y_mask):
        # Get hidden vector

        x1 = x
        for enc in self.enc_list_0:
            x = enc(x, x_mask)

        for enc in self.enc_list_1:
            x1 = enc(x1, x_mask)

        x = x + x1
        for dec in self.dec_list:
            y = dec(y, x, y_mask, x_mask)

        return x, y