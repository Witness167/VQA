from core.model.net_utils import FC, MLP, LayerNorm, MHAtt, FFN, SA

import torch
import torch.nn as nn

# -------------------------------
# ---- AAM Adaptive Activation Module ----
# -------------------------------

class AAM(nn.Module):

    def __init__(self, __C):
        super(AAM, self).__init__()
        self.linear_S_avgPool = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_S_qt = nn.Linear(4, __C.HIDDEN_SIZE)
        self.linear_H_S = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_g_H = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)

        self.sigmoid_S = nn.Sigmoid()
        self.sigmoid_H = nn.Sigmoid()
        self.sigmoid_g = nn.Sigmoid()

    def forward(self, x_update, x_original, G, Question_Type):
        """
        :param x_update:
        :param x_original:
        :param G: question and image global information
        :param Question_Type: question-type information
        :return:
        """
        S = self.sigmoid_S(self.linear_S_avgPool(torch.sum(x_original, dim=1) / 14) * torch.add(G, self.linear_S_qt(Question_Type)))
        H = self.sigmoid_H(self.linear_H_S(S))
        O = self.sigmoid_g(torch.sum(x_update, dim=1) / 14) * H

        return O

# ------------------------------------------------
# ---- AI ----
# ------------------------------------------------

class AI(nn.Module):

    def __init__(self, __C):
        super(AI, self).__init__()

        self.mhatt1 = MHAtt(__C)
        self.mhatt2 = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout3 = nn.Dropout(__C.DROPOUT_R)
        self.norm3 = LayerNorm(__C.HIDDEN_SIZE)

        self.AAM = AAM(__C)

    def forward(self, x, y, x_mask, y_mask, G, Question_Type):

        """
        :param x: image features
        :param y: language features
        """

        x_update = self.norm1(x + self.dropout1(
            self.mhatt1(x, x, x, x_mask)
        ))

        x_update = self.norm2(x_update + self.dropout2(
            self.mhatt2(y, y, x_update, y_mask)
        ))

        g_AA = self.AAM(x_update, x, G, Question_Type).unsqueeze(-2)

        x_output = g_AA * x_update + x

        x_output = self.norm3(x_output + self.dropout3(
            self.ffn(x_output)
        ))

        return x_output

# ------------------------------------------------
# ---- AIN ----
# ------------------------------------------------


class AIN(nn.Module):

    def __init__(self, __C):

        super(AIN, self).__init__()
        self.enc_list = nn.ModuleList([SA(__C) for _ in range(__C.LAYER)])
        self.dec_list = nn.ModuleList([AI(__C) for _ in range(__C.LAYER)])

        self.linear_q_global = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_imgae_global = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_question_type = nn.Linear(__C.HIDDEN_SIZE, 4)
        self.linear_image_ei = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)

    def forward(self, x, y, x_mask, y_mask):
        """
        :param x: language features
        :param y: image features
        """

        for enc in self.enc_list:
            x = enc(x, x_mask)

        # get the global information of question and image
        Q_Global = x[ :, 0, :] + x[ :, -1, :]
        Q_Global = self.linear_q_global(Q_Global)

        I_pooling = torch.sum(y, dim = 1) / 14

        I_ei = torch.tanh(torch.sum(torch.mul(self.linear_image_ei(y), I_pooling.unsqueeze(-2).repeat(1,100,1)), dim = -1))
        I_ei = torch.softmax(I_ei, 1)
        I_Global = self.linear_imgae_global(torch.sum(torch.mul(I_ei.unsqueeze(-1).repeat(1, 1, 512), y) / 100, dim = 1))
        G = torch.tanh(torch.add(Q_Global, I_Global))

        # question type information is the condition of
        Question_Type = self.linear_question_type(Q_Global)

        for i, dec in enumerate(self.dec_list):
            y = dec(y, x, y_mask, x_mask, G, Question_Type)

        return x, y
