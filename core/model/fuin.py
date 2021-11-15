

from core.model.net_utils import LayerNorm, MHAtt, FFN, SA

import torch.nn as nn
import torch
torch.set_printoptions(profile="full")

# ------------------------------------------------
# ---- FAA 细粒度自适应激活模块 ----
# ------------------------------------------------

class FAA(nn.Module):

    def __init__(self, __C):
        super(FAA, self).__init__()
        self.linear_fcd_SA_mul = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_fcd_GA_mul = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, x_SA, x_GA):
        fcd_x_SA = self.sigmoid(self.linear_fcd_SA_mul(x * x_SA)).unsqueeze(-2)
        fcd_x_GA = self.sigmoid(self.linear_fcd_GA_mul(x * x_GA)).unsqueeze(-2)
        """
        torch.Size([64, 100, 1, 512])
        torch.Size([64, 100, 1, 512])
        torch.Size([64, 100, 2, 512])
        fcd_x_sa.size()
        torch.Size([64, 100, 512])
        torch.Size([64, 100, 512])

        """
        fcd_x = torch.softmax(torch.cat((fcd_x_SA, fcd_x_GA), dim=-2), dim=-2) #{bs, n, 2d}
        fcd_x_SA = fcd_x[:, :, 0, :]
        fcd_x_GA = fcd_x[:, :, 1, :]
        return fcd_x_SA, fcd_x_GA

# ------------------------------------------------
# ---- FUI ----
# ------------------------------------------------

class FUI(nn.Module):

    def __init__(self, __C):
        super(FUI, self).__init__()

        self.mhatt_SA = MHAtt(__C)
        self.mhatt_GA = MHAtt(__C)
        self.ffn = FFN(__C)
        self.HIDDEN_SIZE = __C.HIDDEN_SIZE

        self.dropout_fmf = nn.Dropout(__C.DROPOUT_R)
        self.norm_fmf = LayerNorm(__C.HIDDEN_SIZE)
        self.dropout_ffn = nn.Dropout(__C.DROPOUT_R)
        self.norm_ffn = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout_FCD = nn.Dropout(__C.DROPOUT_R)
        self.linear_x_q = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)

        self.faa = FAA(__C)

        input_dim = [__C.HIDDEN_SIZE, __C.HIDDEN_SIZE]
        #self.fusion = MFB(input_dim, __C.HIDDEN_SIZE)

        self.linear_fusion_SA = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE * 8)
        self.linear_fusion_GA = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE * 8)

    def forward(self, x, y, x_mask, y_mask):

        # parallel interaction (PI)
        x_SA = self.mhatt_SA(x, x, x, x_mask)
        x_GA = self.mhatt_GA(y, y, x, y_mask)

        # FAA
        fcd_x_SA, fcd_x_GA = self.faa(x, x_SA, x_GA)

        #fusion
        x_SA = self.linear_fusion_SA(fcd_x_SA * x_SA).reshape([64, 100, 8, -1])
        x_GA = self.linear_fusion_GA(fcd_x_GA * x_GA).reshape([64, 100, 8, -1])

        x_fmf = torch.sum(x_SA * x_GA, dim = -2)

        #x_fmf = self.fusion(x_out)
        #print(x_fmf.size())
        #x_fmf = self.norm_fmf(x + self.dropout_fmf(x_fmf))
        x_output = self.norm_ffn(x_fmf + self.dropout_ffn(self.ffn(x_fmf)))

        return x_output

# ------------------------------------------------
# ---- FUIN ----
# ------------------------------------------------

class FUIN(nn.Module):

    def __init__(self, __C):

        super(FUIN, self).__init__()
        self.enc_list = nn.ModuleList([SA(__C) for _ in range(__C.LAYER)])
        self.dec_list = nn.ModuleList([FUI(__C) for _ in range(__C.LAYER)])

    def forward(self, x, y, x_mask, y_mask):

        for enc in self.enc_list:
            x = enc(x, x_mask)

        for i, dec in enumerate(self.dec_list):
            y = dec(y, x, y_mask, x_mask)

        return x, y