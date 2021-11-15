
import torch
import torch.nn as nn

from core.model.dcn.dense_coattn import DenseCoAttn

# ------------------------------------------------
# ---- DCN  ----
# ------------------------------------------------


class NormalSubLayer(nn.Module):

    def __init__(self, __C, drop=0.1):
        super(NormalSubLayer, self).__init__()
        self.dense_coattn = DenseCoAttn(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE, __C.MULTI_HEAD, num_none=3, dropout=drop)
        self.linears = nn.ModuleList([
            nn.Sequential(
                nn.Linear(__C.HIDDEN_SIZE + __C.HIDDEN_SIZE, __C.HIDDEN_SIZE),
                nn.ReLU(inplace=True),
                nn.Dropout(p=drop),
            ),
            nn.Sequential(
                nn.Linear(__C.HIDDEN_SIZE + __C.HIDDEN_SIZE, __C.HIDDEN_SIZE),
                nn.ReLU(inplace=True),
                nn.Dropout(p=drop),
            )
        ])

    def forward(self, data1, data2, mask1, mask2):
        weighted1, weighted2 = self.dense_coattn(data1, data2, mask1, mask2)
        data1 = data1 + self.linears[0](torch.cat([data1, weighted2], dim=2))
        data2 = data2 + self.linears[1](torch.cat([data2, weighted1], dim=2))

        return data1, data2

class DCN(nn.Module):
    def __init__(self, __C):

        super(DCN, self).__init__()
        self.dcn_layers = nn.ModuleList([NormalSubLayer(__C) for _ in range(__C.LAYER)])
        print("__C.LAYER--:" + str(__C.LAYER))

    def forward(self, x, y, x_mask, y_mask):

        for dense_coatten in self.dcn_layers:
            x, y = dense_coatten(x, y, x_mask, y_mask)

        return x, y