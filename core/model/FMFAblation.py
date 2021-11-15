import torch
import torch.nn as nn
from .FMF import MHAtt, FFN, LayerNorm, SA, FMF_FOR_ED_A, fcd, qkv_attention, cosine_similarity

#-------------------------------------------
#----------  FMF stacking alation
#-------------------------------------------

class FMF(nn.Module):
    """
    todo
    """

    def __init__(self, __C):
        super(FMF, self).__init__()

        self.mhatt1 = MHAtt(__C)
        self.mhatt2 = MHAtt(__C)
        self.ffn = FFN(__C)
        self.HIDDEN_SIZE = __C.HIDDEN_SIZE

        self.dropout_x_SA = nn.Dropout(__C.DROPOUT_R)
        self.norm_x_SA = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout_x_GA = nn.Dropout(__C.DROPOUT_R)
        self.norm_x_GA = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout_y_SA = nn.Dropout(__C.DROPOUT_R)
        self.norm_y_SA = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout_y_GA = nn.Dropout(__C.DROPOUT_R)
        self.norm_y_GA = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout_x = nn.Dropout(__C.DROPOUT_R)
        self.norm_x = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout_y = nn.Dropout(__C.DROPOUT_R)
        self.norm_y = LayerNorm(__C.HIDDEN_SIZE)

        #fcd input x y
        self.linear_v_x = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_k_x = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q_x = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)

        self.linear_v_y = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_k_y = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q_y = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)



    def forward(self, x, y, x_mask, y_mask):
        """
        take image as example, x=image, y=text, contain sa and ga
        :param x:
        :param y:
        :param x_mask:
        :param y_mask:
        :return:
        """

        #SA and GA
        x_SA = self.norm_x_SA(x + self.dropout_x_SA(
            self.mhatt1(x, x, x, x_mask)
        ))
        x_GA = self.norm_x_GA(x + self.dropout_x_GA(
            self.mhatt2(y, y, x, y_mask)
        ))
        y_SA = self.norm_y_SA(y + self.dropout_y_SA(
            self.mhatt1(y, y, y, y_mask)
        ))
        y_GA = self.norm_y_GA(y + self.dropout_y_GA(
            self.mhatt2(x, x, y, x_mask)
        ))

        #fcd input x y
        x_fcd_v = self.linear_v_x(x)
        x_fcd_k = self.linear_k_x(x)
        x_fcd_q = self.linear_q_x(x)

        y_fcd_v = self.linear_v_y(x)
        y_fcd_k = self.linear_k_y(x)
        y_fcd_q = self.linear_q_y(x)

        #FCD
        # (batch_size, l)
        fcd_x_SA, fcd_x_GA, fcd_y_SA, fcd_y_GA = fcd(x_fcd_v, x_fcd_k, x_fcd_q, y_fcd_v, y_fcd_k, y_fcd_q)

        #MF
        # (batch_size, l, dim)
        x = torch.add(
            torch.mul(
                x_SA, fcd_x_SA.unsqueeze(-1)
            ),
            torch.mul(
                x_GA, fcd_x_GA.unsqueeze(-1)
            )
        )

        y = torch.add(
            torch.mul(
                y_SA, fcd_y_SA.unsqueeze(-1)
            ),
            torch.mul(
                y_GA, fcd_y_GA.unsqueeze(-1)
            )
        )

        x = self.norm_x(x + self.dropout_x(
            self.ffn(x)
        ))
        y = self.norm_y(y + self.dropout_y(
            self.ffn(y)
        ))
        return x, y

class FMF_STACK(nn.Module):
    def __init__(self, __C):
        super(FMF_STACK, self).__init__()

        self.stack_list = nn.ModuleList([FMF(__C) for _ in range(__C.LAYER)])

    def forward(self, x, y, x_mask, y_mask):
        for stack in self.stack_list:
            x, y = stack(x, y, x_mask, y_mask)

        return x, y

#-------------------------------------------
#----------  FMF only self and self-cross and cross  ablation
#-------------------------------------------

class FCD_OS(nn.Module):

    def __init__(self, __C):
        super(FCD_OS, self).__init__()
        self.linear_fcd_SA_mul = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_fcd_GA_mul = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.sigmoid = nn.Sigmoid()
        self.FCD = __C.FCD

    def forward(self, x, x_SA):
        fcd_x_SA = self.sigmoid(self.linear_fcd_SA_mul(x * x_SA)).unsqueeze(-2)
        fcd_x_GA = self.sigmoid(self.linear_fcd_GA_mul(x * x_SA)).unsqueeze(-2)
        """
        torch.Size([64, 100, 1, 512])
        torch.Size([64, 100, 1, 512])
        torch.Size([64, 100, 2, 512])
        fcd_x_sa.size()
        torch.Size([64, 100, 512])
        torch.Size([64, 100, 512])

        """
        fcd_x = torch.softmax(torch.cat((fcd_x_SA, fcd_x_GA), dim=-2), dim=-2)  # {bs, n, 2d}
        fcd_x_SA = fcd_x[:, :, 0, :]
        fcd_x_GA = fcd_x[:, :, 1, :]
        return fcd_x_SA, fcd_x_GA

class FCD_OC(nn.Module):

    def __init__(self, __C):
        super(FCD_OC, self).__init__()
        self.linear_fcd_SA_mul = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_fcd_GA_mul = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.sigmoid = nn.Sigmoid()
        self.FCD = __C.FCD

    def forward(self, x, x_GA):
        fcd_x_SA = self.sigmoid(self.linear_fcd_SA_mul(x * x_GA)).unsqueeze(-2)
        fcd_x_GA = self.sigmoid(self.linear_fcd_GA_mul(x * x_GA)).unsqueeze(-2)
        """
        torch.Size([64, 100, 1, 512])
        torch.Size([64, 100, 1, 512])
        torch.Size([64, 100, 2, 512])
        fcd_x_sa.size()
        torch.Size([64, 100, 512])
        torch.Size([64, 100, 512])

        """
        fcd_x = torch.softmax(torch.cat((fcd_x_SA, fcd_x_GA), dim=-2), dim=-2)  # {bs, n, 2d}
        fcd_x_SA = fcd_x[:, :, 0, :]
        fcd_x_GA = fcd_x[:, :, 1, :]
        return fcd_x_SA, fcd_x_GA

class FMF_FOR_ED_OC(nn.Module):

    def __init__(self, __C):
        super(FMF_FOR_ED_OC, self).__init__()

        self.mhatt_SA = MHAtt(__C)
        self.mhatt_GA = MHAtt(__C)
        self.ffn = FFN(__C)
        self.HIDDEN_SIZE = __C.HIDDEN_SIZE

        self.dropout_fmf = nn.Dropout(__C.DROPOUT_R)
        self.norm_fmf = LayerNorm(__C.HIDDEN_SIZE)
        self.dropout_ffn = nn.Dropout(__C.DROPOUT_R)
        self.norm_ffn = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout_FCD = nn.Dropout(__C.DROPOUT_R)
        self.fcd = FCD_OC(__C)

    def forward(self, x, y, x_mask, y_mask):
        x_SA = self.mhatt_SA(x, x, x, x_mask)
        x_GA = self.mhatt_GA(y, y, x, y_mask)
        fcd_x_SA, fcd_x_GA = self.fcd(x, x_GA)
        x_fmf = fcd_x_SA * x_SA + fcd_x_GA * x_GA
        x_fmf = self.norm_fmf(x + self.dropout_fmf(x_fmf))
        x_output = self.norm_ffn(x_fmf + self.dropout_ffn(self.ffn(x_fmf)))
        return x_output

class FMF_FOR_ED_OS(nn.Module):

    def __init__(self, __C):
        super(FMF_FOR_ED_OS, self).__init__()

        self.mhatt_SA = MHAtt(__C)
        self.mhatt_GA = MHAtt(__C)
        self.ffn = FFN(__C)
        self.HIDDEN_SIZE = __C.HIDDEN_SIZE

        self.dropout_fmf = nn.Dropout(__C.DROPOUT_R)
        self.norm_fmf = LayerNorm(__C.HIDDEN_SIZE)
        self.dropout_ffn = nn.Dropout(__C.DROPOUT_R)
        self.norm_ffn = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout_FCD = nn.Dropout(__C.DROPOUT_R)
        self.fcd = FCD_OS(__C)

    def forward(self, x, y, x_mask, y_mask):
        x_SA = self.mhatt_SA(x, x, x, x_mask)
        x_GA = self.mhatt_GA(y, y, x, y_mask)
        fcd_x_SA, fcd_x_GA = self.fcd(x, x_SA)
        x_fmf = fcd_x_SA * x_SA + fcd_x_GA * x_GA
        x_fmf = self.norm_fmf(x + self.dropout_fmf(x_fmf))
        x_output = self.norm_ffn(x_fmf + self.dropout_ffn(self.ffn(x_fmf)))
        return x_output

class FMF_ED_OC(nn.Module):
    def __init__(self, __C):
        super(FMF_ED_OC, self).__init__()

        self.enc_list = nn.ModuleList([SA(__C) for _ in range(__C.LAYER)])
        self.dec_list = nn.ModuleList([FMF_FOR_ED_OC(__C) for _ in range(__C.LAYER)])

    def forward(self, x, y, x_mask, y_mask):
        for enc in self.enc_list:
            x = enc(x, x_mask)

        for dec in self.dec_list:
            y = dec(y, x, y_mask, x_mask)

        return x, y

class FMF_ED_OS(nn.Module):
    def __init__(self, __C):
        super(FMF_ED_OS, self).__init__()

        self.enc_list = nn.ModuleList([SA(__C) for _ in range(__C.LAYER)])
        self.dec_list = nn.ModuleList([FMF_FOR_ED_OS(__C) for _ in range(__C.LAYER)])

    def forward(self, x, y, x_mask, y_mask):
        for enc in self.enc_list:
            x = enc(x, x_mask)

        for dec in self.dec_list:
            y = dec(y, x, y_mask, x_mask)

        return x, y



#-------------------------------------------
#----------  Unbalance ablation
#-------------------------------------------

class USGA(nn.Module):
    """
    unbalanced means SA and GA are not equal
    """
    def __init__(self, __C):
        super(USGA, self).__init__()

        #todo
        #个数问题
        self.mhatt1 = MHAtt(__C)
        self.mhatt2 = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout_x_SA = nn.Dropout(__C.DROPOUT_R)
        self.norm_x_SA = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout_x_GA = nn.Dropout(__C.DROPOUT_R)
        self.norm_x_GA = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout_y_SA = nn.Dropout(__C.DROPOUT_R)
        self.norm_y_SA = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout_y_GA = nn.Dropout(__C.DROPOUT_R)
        self.norm_y_GA = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout_x = nn.Dropout(__C.DROPOUT_R)
        self.norm_x = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout_y = nn.Dropout(__C.DROPOUT_R)
        self.norm_y = LayerNorm(__C.HIDDEN_SIZE)

        self.SA_W = __C.SA_W
        self.GA_W = 1 - __C.SA_W

    def forward(self, x, y, x_mask, y_mask):

        x_SA = self.norm_x_SA(x + self.dropout_x_SA(
            self.mhatt1(x, x, x, x_mask)
        ))

        x_GA = self.norm_x_GA(x + self.dropout_x_GA(
            self.mhatt2(y, y, x, y_mask)
        ))

        y_SA = self.norm_y_SA(y + self.dropout_y_SA(
            self.mhatt1(y, y, y, y_mask)
        ))

        y_GA = self.norm_y_GA(y + self.dropout_y_GA(
            self.mhatt2(x, x, y, x_mask)
        ))

        x = torch.add(self.SA_W * x_SA, self.GA_W * x_GA)

        x = self.norm_x(x + self.dropout_x(
            self.ffn(x)
        ))

        y = torch.add(self.SA_W * y_SA, self.GA_W * y_GA)

        y = self.norm_y(y + self.dropout_y(
            self.ffn(y)
        ))
        return x, y

class UMF_STACK(nn.Module):
    def __init__(self, __C):
        super(UMF_STACK, self).__init__()
        self.stack_list = nn.ModuleList([USGA(__C) for _ in range(__C.LAYER)])

    def forward(self, x, y, x_mask, y_mask):
        for stack in self.stack_list:
            x, y = stack(x, y, x_mask, y_mask)
        return x, y

class USGA_ED(nn.Module):
    def __init__(self, __C):
        super(USGA_ED, self).__init__()

        self.mhatt_SA = MHAtt(__C)
        self.mhatt_GA = MHAtt(__C)
        self.ffn = FFN(__C)
        self.HIDDEN_SIZE = __C.HIDDEN_SIZE

        self.dropout_fmf = nn.Dropout(__C.DROPOUT_R)
        self.norm_fmf = LayerNorm(__C.HIDDEN_SIZE)
        self.dropout_ffn = nn.Dropout(__C.DROPOUT_R)
        self.norm_ffn = LayerNorm(__C.HIDDEN_SIZE)
        self.dropout_FCD = nn.Dropout(__C.DROPOUT_R)

        #different weight of SA and GA
        self.SA_W = __C.SA_W
        self.GA_W = 1 - __C.SA_W

    def forward(self, x, y, x_mask, y_mask):
        x_fmf = self.norm_fmf(x + self.dropout_fmf(self.SA_W * self.mhatt_SA(x, x, x, x_mask) + self.GA_W * self.mhatt_GA(y, y, x, y_mask)))

        x_output = self.norm_ffn(x_fmf + self.dropout_ffn(self.ffn(x_fmf)))
        return x_output

class UMF_ED(nn.Module):
    def __init__(self, __C):
        super(UMF_ED, self).__init__()

        self.enc_list = nn.ModuleList([SA(__C) for _ in range(__C.LAYER)])
        self.dec_list = nn.ModuleList([USGA_ED(__C) for _ in range(__C.LAYER)])
        print(__C.LAYER)

    def forward(self, x, y, x_mask, y_mask):
        # Get hidden vector
        for enc in self.enc_list:
            x = enc(x, x_mask)
        for dec in self.dec_list:
            y = dec(y, x, y_mask, x_mask)
        return x, y

