from core.model.net_utils import FC, MLP, LayerNorm, SA, SGA

import torch.nn as nn
import torch.nn.functional as F
import torch, math
import copy



# ------------------------------------------------
# ----botton up (2018 CVPR)----
# ----question guided image feature
# ------------------------------------------------


class BottomUp(nn.Module):
    def __init__(self, __C):
        super(BottomUp, self).__init__()
        self.__C = __C

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE + 2048,
            mid_size=__C.FLAT_MLP_SIZE,
            out_size=__C.FLAT_GLIMPSES,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            2048 * __C.FLAT_GLIMPSES,
            __C.HIDDEN_SIZE
        )

    def forward(self, x, y):
        """

        :param x: language {batch_size, 14, 512}
        :param y: image {batch_size, 100, 2048}
        :param x_mask:
        :param y_mask:
        :return: language {batch, 512}  image {batch, 512}
        """

        x = torch.sum(x, dim=1) #{b, 512}
        w_x = x.unsqueeze(1).expand(-1, 100, -1) # torch.Size([64, 100, 512])
        att = self.mlp(torch.cat((w_x, y), dim=2))
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.__C.FLAT_GLIMPSES):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * y, dim=1)
            )

        y = torch.cat(att_list, dim=1)
        y = self.linear_merge(y)

        return x, y