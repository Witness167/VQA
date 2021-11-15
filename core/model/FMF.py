

from core.model.net_utils import FC, MLP, LayerNorm, MHAtt, FFN, SA

import torch.nn as nn
import torch.nn.functional as F
import torch, math
import copy
import numpy as np
torch.set_printoptions(profile="full")


from math import sqrt

def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


class MHAtt_for_FMF(nn.Module):
    """
    使得可以实现SA和GA共享查询向量
    """
    def  __init__(self, __C):
        super(MHAtt_for_FMF, self).__init__()
        self.__C = __C

        self.linear_merge = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)

        self.dropout = nn.Dropout(__C.DROPOUT_R)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = v.view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            self.__C.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        k = k.view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            self.__C.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        q = q.view(
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

        atted = self.linear_merge(atted)

        return atted

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

def qkv_attention(query, key, value, dropout=None, mask=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / sqrt(d_k)
    if mask is not None:
        #{[64, 1, 1, 14]}, size=[64, 100, 14] for squeeze(1)
        # becouse fcd without multi-head
        scores.data.masked_fill_(mask.squeeze(1), -65504.0)  #{[64, 1, 1, 14]}, size=[64, 100, 14]

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn

def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()

def fcd(x, x_attended_SA, x_attended_GA):
    fcd_x_SA = cosine_similarity(x, x_attended_SA, dim=2).unsqueeze(-1)
    fcd_x_GA = cosine_similarity(x, x_attended_GA, dim=2).unsqueeze(-1)

    fcd_x = torch.softmax(torch.cat((fcd_x_SA, fcd_x_GA), dim=-1), dim=-1)

    fcd_x_SA = fcd_x[:,:,0].squeeze()
    fcd_x_GA = fcd_x[:,:,-1].squeeze()
    return fcd_x_SA, fcd_x_GA

class FCD(nn.Module):

    def __init__(self, __C):
        super(FCD, self).__init__()
        self.linear_fcd_SA_cat = nn.Linear(__C.HIDDEN_SIZE * 2, __C.HIDDEN_SIZE)
        self.linear_fcd_GA_cat = nn.Linear(__C.HIDDEN_SIZE * 2, __C.HIDDEN_SIZE)
        self.linear_fcd_SA_mul = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_fcd_GA_mul = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.sigmoid = nn.Sigmoid()
        self.FCD = __C.FCD

    def forward(self, x, x_SA, x_GA):
        if self.FCD == "cat":
            fcd_x_SA = self.sigmoid(self.linear_fcd_SA_cat(torch.cat((x * x_SA, x * x_GA), dim=-1))).unsqueeze(-2)  #{bs, n, d}
            fcd_x_GA = self.sigmoid(self.linear_fcd_GA_cat(torch.cat((x * x_SA, x * x_GA), dim=-1))).unsqueeze(-2)  #{bs, n, d}
        elif self.FCD == "mul":
            fcd_x_SA = self.sigmoid(self.linear_fcd_SA_mul(x * x_SA)).unsqueeze(-2)
            fcd_x_GA = self.sigmoid(self.linear_fcd_GA_mul(x * x_GA)).unsqueeze(-2)
        elif self.FCD == "o":
            fcd_x_SA = self.sigmoid(x * x_SA).unsqueeze(-2)
            fcd_x_GA = self.sigmoid(x * x_GA).unsqueeze(-2)
        else:
            print("fcd type wrong!!!!")
            exit()
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
# ---- FMF ----
# ------------------------------------------------

class FMF_FOR_ED_A(nn.Module):

    def __init__(self, __C):
        super(FMF_FOR_ED_A, self).__init__()

        self.mhatt_SA = MHAtt(__C)
        self.mhatt_GA = MHAtt(__C)
        self.ffn = FFN(__C)
        self.HIDDEN_SIZE = __C.HIDDEN_SIZE

        self.dropout_fmf = nn.Dropout(__C.DROPOUT_R)
        self.norm_fmf = LayerNorm(__C.HIDDEN_SIZE)
        self.dropout_ffn = nn.Dropout(__C.DROPOUT_R)
        self.norm_ffn = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout_FCD = nn.Dropout(__C.DROPOUT_R)
        self.fcd = FCD(__C)
        self.linear_x_q = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)

    def forward(self, x, y, x_mask, y_mask):

        x_SA, atten_map_SA = self.mhatt_SA(x, x, x, x_mask)
        x_GA, atten_map_GA = self.mhatt_GA(y, y, x, y_mask)
        fcd_x_SA, fcd_x_GA = self.fcd(x, x_SA, x_GA)
        x_fmf = fcd_x_SA * x_SA + fcd_x_GA * x_GA
        x_fmf = self.norm_fmf(x + self.dropout_fmf(x_fmf))
        x_output = self.norm_ffn(x_fmf + self.dropout_ffn(self.ffn(x_fmf)))
        return x_output, atten_map_SA, atten_map_GA, torch.sum(fcd_x_SA, dim=-1) / 512, torch.sum(fcd_x_GA, dim=-1) / 512

class FMF_FOR_ED_B(nn.Module):
    """
    FMF layer for encoder decoder
    a simple layer
    """

    def __init__(self, __C):
        super(FMF_FOR_ED_B, self).__init__()

        self.mhatt_SA = MHAtt_for_FMF(__C)
        self.mhatt_GA = MHAtt_for_FMF(__C)
        self.ffn = FFN(__C)
        self.HIDDEN_SIZE = __C.HIDDEN_SIZE

        self.dropout_fmf = nn.Dropout(__C.DROPOUT_R)
        self.norm_fmf = LayerNorm(__C.HIDDEN_SIZE)
        self.dropout_ffn = nn.Dropout(__C.DROPOUT_R)
        self.norm_ffn = LayerNorm(__C.HIDDEN_SIZE)

        # fcd input x y
        self.linear_v_x = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_k_x = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q_x = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)

        self.linear_v_y = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_k_y = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q_y = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)

        # sga input x y
        self.linear_v_x_sga = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_k_x_sga = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q_x_sga = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)

        self.linear_v_y_sga = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_k_y_sga = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q_y_sga = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)

        self.dropout_FCD = nn.Dropout(__C.DROPOUT_R)

    def forward(self, x, y, x_mask, y_mask):

        # fcd input x y
        # x_fcd_v = self.linear_v_x(x)
        # x_fcd_k = self.linear_k_x(x)
        # x_fcd_q = self.linear_q_x(x)
        #
        # y_fcd_v = self.linear_v_y(y)
        # y_fcd_k = self.linear_k_y(y)
        # y_fcd_q = self.linear_q_y(y)

        # sga input x y
        x_fcd_v_sga = self.linear_v_x_sga(x)
        x_fcd_k_sga = self.linear_k_x_sga(x)
        x_fcd_q_sga = self.linear_q_x_sga(x)

        y_fcd_v_sga = self.linear_v_y_sga(y)
        y_fcd_k_sga = self.linear_k_y_sga(y)
        y_fcd_q_sga = self.linear_q_y_sga(y)

        x_SA = self.mhatt_SA(x_fcd_v_sga, x_fcd_k_sga, x_fcd_q_sga, x_mask)
        x_GA = self.mhatt_GA(y_fcd_v_sga, y_fcd_k_sga, x_fcd_q_sga, y_mask)

        x_fmf = self.norm_fmf(x + self.dropout_fmf(x_SA + x_GA))

        # FCD
        # (batch_size, l)
        #fcd_x_SA, fcd_x_GA, fcd_y_SA, fcd_y_GA = fcd(x_fcd_v, x_fcd_k, x_fcd_q, y_fcd_v, y_fcd_k, y_fcd_q, dropout=self.dropout_FCD, x_mask=x_mask, y_mask=y_mask)

        #MF
        # (batch_size, l, dim)
        # x_FMF = torch.add(
        #     torch.mul(
        #         x_SA, fcd_x_SA.unsqueeze(-1)
        #     ),
        #     torch.mul(
        #         x_GA, fcd_x_GA.unsqueeze(-1)
        #     )
        # )

        x_output = self.norm_ffn(x_fmf + self.dropout_ffn(self.ffn(x_fmf)))
        return x_output

class FMF_FOR_ED_C(nn.Module):
    """
    FMF layer for encoder decoder
    a simple layer
    """

    def __init__(self, __C):
        super(FMF_FOR_ED_C, self).__init__()

        self.mhatt1 = MHAtt_for_FMF(__C)
        self.mhatt2 = MHAtt_for_FMF(__C)
        self.ffn = FFN(__C)
        self.HIDDEN_SIZE = __C.HIDDEN_SIZE

        self.dropout_fmf = nn.Dropout(__C.DROPOUT_R)
        self.norm_fmf = LayerNorm(__C.HIDDEN_SIZE)
        self.dropout_ffn = nn.Dropout(__C.DROPOUT_R)
        self.norm_ffn = LayerNorm(__C.HIDDEN_SIZE)

        # fcd input x y
        self.linear_v = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_k = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)

        self.FCD_TYPE = __C.FCD_TYPE

        self.dropout_FCD = nn.Dropout(__C.DROPOUT_R)

    def forward(self, x, y, x_mask, y_mask):

        # fcd input x y
        x_fcd_v = self.linear_v(x)
        x_fcd_k = self.linear_k(x)
        x_fcd_q = self.linear_q(x)

        y_fcd_v = self.linear_v(y)
        y_fcd_k = self.linear_k(y)
        y_fcd_q = self.linear_q(y)

        x_SA = self.mhatt1(x_fcd_v, x_fcd_k, x_fcd_q, x_mask)
        x_GA = self.mhatt2(y_fcd_v, y_fcd_k, x_fcd_q, y_mask)
        x_fmf = self.norm_fmf(x + self.dropout_fmf(x_SA + x_GA))

        # FCD
        # (batch_size, l)
        #fcd_x_SA, fcd_x_GA, fcd_y_SA, fcd_y_GA = fcd(x_fcd_v, x_fcd_k, x_fcd_q, y_fcd_v, y_fcd_k, y_fcd_q, dropout=self.dropout_FCD, x_mask=x_mask, y_mask=y_mask)

        #MF
        # (batch_size, l, dim)
        # x_FMF = torch.add(
        #     torch.mul(
        #         x_SA, fcd_x_SA.unsqueeze(-1)
        #     ),
        #     torch.mul(
        #         x_GA, fcd_x_GA.unsqueeze(-1)
        #     )
        # )

        x_output = self.norm_ffn(x_fmf + self.dropout_ffn(self.ffn(x_fmf)))
        return x_output

class FMF_FOR_ED_D(nn.Module):

    def __init__(self, __C):
        super(FMF_FOR_ED_D, self).__init__()

        self.mhatt_fmf = MHAtt_for_FMF(__C)
        self.ffn = FFN(__C)

        self.dropout_fmf = nn.Dropout(__C.DROPOUT_R)
        self.norm_fmf = LayerNorm(__C.HIDDEN_SIZE)
        self.dropout_ffn = nn.Dropout(__C.DROPOUT_R)
        self.norm_ffn = LayerNorm(__C.HIDDEN_SIZE)

        # fcd input x y
        self.linear_v = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_k = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)

    def forward(self, x, y, x_mask, y_mask):

        # fcd input x y
        x_fcd_v = self.linear_v(x)
        x_fcd_k = self.linear_k(x)
        x_fcd_q = self.linear_q(x)

        y_fcd_v = self.linear_v(y)
        y_fcd_k = self.linear_k(y)
        y_fcd_q = self.linear_q(y)

        v = torch.cat((x_fcd_v, y_fcd_v), dim=-2)
        k = torch.cat((x_fcd_k, y_fcd_k), dim=-2)
        q = torch.cat((x_fcd_q, y_fcd_q), dim=-2)
        xy = torch.cat((x, y), dim=-2)

        mask = (torch.sum(torch.abs(v),dim=-1) == 0).unsqueeze(1).unsqueeze(2)
        x_SA = self.mhatt_fmf(v, k, q, mask)
        x_fmf = self.norm_fmf(xy + self.dropout_fmf(x_SA))

        # FCD
        # (batch_size, l)
        #fcd_x_SA, fcd_x_GA, fcd_y_SA, fcd_y_GA = fcd(x_fcd_v, x_fcd_k, x_fcd_q, y_fcd_v, y_fcd_k, y_fcd_q, dropout=self.dropout_FCD, x_mask=x_mask, y_mask=y_mask)

        #MF
        # (batch_size, l, dim)
        # x_FMF = torch.add(
        #     torch.mul(
        #         x_SA, fcd_x_SA.unsqueeze(-1)
        #     ),
        #     torch.mul(
        #         x_GA, fcd_x_GA.unsqueeze(-1)
        #     )
        # )

        x_output = self.norm_ffn(x_fmf + self.dropout_ffn(self.ffn(x_fmf)))
        x_output = x_output[:,0:100,:]
        return x_output

class FMF_ED(nn.Module):
    def __init__(self, __C):
        super(FMF_ED, self).__init__()
        #language guided to learn image feature
        self.enc_list = nn.ModuleList([SA(__C) for _ in range(__C.LAYER)])
        if __C.SGT == "A":
            self.dec_list = nn.ModuleList([FMF_FOR_ED_A(__C) for _ in range(__C.LAYER)])
        elif __C.SGT == "B":
            self.dec_list = nn.ModuleList([FMF_FOR_ED_B(__C) for _ in range(__C.LAYER)])
        elif __C.SGT == "C":
            self.dec_list = nn.ModuleList([FMF_FOR_ED_C(__C) for _ in range(__C.LAYER)])
        elif __C.SGT == "D":
            self.dec_list = nn.ModuleList([FMF_FOR_ED_D(__C) for _ in range(__C.LAYER)])
        print("-------------__C.SGT:" + str(__C.SGT))

    def forward(self, x, y, x_mask, y_mask):
        atten_map_SA_layers = []
        atten_map_GA_layers = []
        fcd_x_SA_layers = []
        fcd_x_GA_layers = []

        atten_map_show_layers = [0, 2, 5]

        for enc in self.enc_list:
            x = enc(x, x_mask)

        for i, dec in enumerate(self.dec_list):
            y, atten_map_SA, atten_map_GA, fcd_x_SA, fcd_x_GA = dec(y, x, y_mask, x_mask)

            if i in atten_map_show_layers:
                atten_map_SA_layers.append(atten_map_SA.cpu().data.numpy())
                atten_map_GA_layers.append(atten_map_GA.cpu().data.numpy())

            fcd_x_SA_layers.append(fcd_x_SA.cpu().data.numpy())
            fcd_x_GA_layers.append(fcd_x_GA.cpu().data.numpy())

        return x, y, np.array(atten_map_SA_layers), np.array(atten_map_GA_layers), np.array(fcd_x_SA_layers), np.array(fcd_x_GA_layers)

"""
all 67.37  
hidden size 768 
seed 15036332
ANSWER_PATH-->{'train': './datasets/vqa/v2_mscoco_train2014_annotations.json', 'val': './datasets/vqa/v2_mscoco_val2014_annotations.json', 'vg': './datasets/vqa/VG_annotations.json'}
BACKBONE-->FMF-ED
BATCH_SIZE-->64
CACHE_PATH-->./results/cache/
CKPTS_PATH-->./ckpts/
CKPT_EPOCH-->13
CKPT_PATH-->None
CKPT_VERSION-->small
DATASET_PATH-->./datasets/vqa/
DEVICES-->[0]
DG-->96
DROPOUT_R-->0.1
EVAL_BATCH_SIZE-->32
EVAL_EVERY_EPOCH-->True
FCD_TYPE-->SAC
FEATURE_PATH-->./datasets/coco_extract/
FF_SIZE-->3072
FLAT_GLIMPSES-->1
FLAT_MLP_SIZE-->512
FLAT_OUT_SIZE-->1024
GPU-->3
GRAD_ACCU_STEPS-->1
GRAD_NORM_CLIP-->-1
HIDDEN_SIZE-->768
HIDDEN_SIZE_HEAD-->96
IMG_FEAT_PAD_SIZE-->100
IMG_FEAT_PATH-->{'train': './datasets/coco_extract/train2014/', 'val': './datasets/coco_extract/val2014/', 'test': './datasets/coco_extract/test2015/'}
IMG_FEAT_SIZE-->2048
LAYER-->6
LOG_PATH-->./results/log/
LR_BASE-->0.0001
LR_DECAY_LIST-->[10, 12]
LR_DECAY_R-->0.2
MAX_EPOCH-->13
MAX_TOKEN-->14
MODEL-->small
MULTI_HEAD-->8
NUM_WORKERS-->16
N_GPU-->1
OPT_BETAS-->(0.9, 0.98)
OPT_EPS-->1e-09
PIN_MEM-->True
PRED_PATH-->./results/pred/
PRELOAD-->False
QUESTION_PATH-->{'train': './datasets/vqa/v2_OpenEnded_mscoco_train2014_questions.json', 'val': './datasets/vqa/v2_OpenEnded_mscoco_val2014_questions.json', 'test': './datasets/vqa/v2_OpenEnded_mscoco_test2015_questions.json', 'vg': './datasets/vqa/VG_questions.json'}
RESULT_PATH-->./results/result_test/
RESUME-->False
RUN_MODE-->train
SA_W-->0.5
SEED-->15036332
SGT-->A
SHUFFLE_MODE-->external
SPLIT-->{'train': 'train', 'val': 'val', 'test': 'test'}
SUB_BATCH_SIZE-->64
TEST_SAVE_PRED-->False
TRAIN_SPLIT-->train
USE_GLOVE-->True
VERBOSE-->True
VERSION-->FMF-ED-6-val-A-wff-hs768
WORD_EMBED_SIZE-->300
"""