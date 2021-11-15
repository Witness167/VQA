##########################


import math
import numpy as np

import torch
import torch.nn as nn
from core.model.net_utils import MHAtt, FFN, LayerNorm, SA

# ------------------------------------------------
# ---- DFAF : DSA + GA (2019 CVPR)----
# ------------------------------------------------
"""
layer = 6
image feature 2048
question feature 1280 
fused feature 1024
latent d 512 multi-head 8   d/h=64
beta1 2 = 0.9 0.98
learning rate min(2.5et-5, 1e-4) 
decay after 10 epochs 
decay rate 0.2 each 2 epochs
train 13 epochs 
batch size 64
"""

class DFAF(nn.Module):
    """
    Single Block Inter-/Intra-modality stack multiple times
    """
    def __init__(self, __C, drop=0.1):
        super(DFAF, self).__init__()

        self.interBlock       = InterModalityUpdate(__C)
        self.intraBlock       = DyIntraModalityUpdate(__C)

        self.drop = nn.Dropout(drop)
        self.num_block = __C.LAYER

    def forward(self, x, y, x_mask, y_mask):
        """
            v: visual feature      [batch, num_obj, feat_size]
            q: question            [batch, max_len, feat_size]
            v_mask                 [batch, num_obj]
            q_mask                 [batch, max_len]
        """

        for i in range(self.num_block):
            x, y = self.interBlock(x, y, x_mask, y_mask)
            x, y = self.intraBlock(x, y, x_mask, y_mask)

        return x, y

class DyIntraModalityUpdate(nn.Module):
    def __init__(self, __C):
        super(DyIntraModalityUpdate, self).__init__()

        self.mhatt1 = MHAtt(__C)
        self.mhatt2 = MHAtt(__C)

        self.dropout_x = nn.Dropout(__C.DROPOUT_R)
        self.norm_x = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout_y = nn.Dropout(__C.DROPOUT_R)
        self.norm_y = LayerNorm(__C.HIDDEN_SIZE)

        self.x2y_linear = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.y2x_linear = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        self.batch_size = __C.BATCH_SIZE
        self.HIDDEN_SIZE = __C.HIDDEN_SIZE

        self.x_output = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.y_output = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)

        self.SA = SA(__C)

    def forward(self, x, y, x_mask, y_mask):
        # average pooling of language and image
        x_mean = x.sum(1)
        y_mean = y.sum(1)

        #conditioned gating vector
        x2y = self.sigmoid(self.x2y_linear(self.dropout_x(self.relu(x_mean)))).unsqueeze(1) # [batch, 1, feat_size]
        y2x = self.sigmoid(self.y2x_linear(self.dropout_y(self.relu(y_mean)))).unsqueeze(1)

        x = (1 + y2x) * x
        y = (1 + x2y) * y

        update_x = self.SA(x, x_mask)
        update_y = self.SA(y, y_mask)

        updated_x = self.x_output(self.dropout_x(x + update_x))
        updated_y = self.x_output(self.dropout_x(y + update_y))

        return updated_x, updated_y

class InterModalityUpdate(nn.Module):
    def __init__(self, __C):
        super(InterModalityUpdate, self).__init__()

        self.mhatt1 = MHAtt(__C)
        self.mhatt2 = MHAtt(__C)

        self.dropout_x = nn.Dropout(__C.DROPOUT_R)

        self.dropout_y = nn.Dropout(__C.DROPOUT_R)

        self.x_output = nn.Linear(__C.HIDDEN_SIZE + __C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.y_output = nn.Linear(__C.HIDDEN_SIZE + __C.HIDDEN_SIZE, __C.HIDDEN_SIZE)

    def forward(self, x, y, x_mask, y_mask):

        cat_x = torch.cat((x, self.dropout_y(self.mhatt1(y, y, x, y_mask))), dim=2)
        cat_y = torch.cat((y, self.dropout_x(self.mhatt2(x, x, y, x_mask))), dim=2)

        x = self.x_output(self.dropout_x(cat_x))
        y = self.y_output(self.dropout_y(cat_y))

        return x, y