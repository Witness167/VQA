# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from core.model.net_utils import FC, MLP, LayerNorm, AttFlat

from core.model.dcn.DCN import DCN
from core.model.DFAF import DFAF
from core.model.MUAN import MUAN
from core.model.MCAN import MCAN
from core.model.SelRes import SelRes
from core.model.MEDAN import MEDAN
from core.model.MCAoAN import MCAoAN
from core.model.mca import MCA_ED

from core.model.ain import AIN
from core.model.fuin import FUIN

import torch.nn as nn
import torch.nn.functional as F
import torch


# -------------------------
# ---- Main Model ----
# -------------------------

class Net(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size, answer_size):
        super(Net, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=__C.WORD_EMBED_SIZE
        )

        # Loading the GloVe embedding weights
        if __C.USE_GLOVE:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.lstm = nn.LSTM(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        self.img_feat_linear = nn.Linear(
            __C.IMG_FEAT_SIZE,
            __C.HIDDEN_SIZE
        )

        # ------------------------
        # ---- Backbone Params ----
        # ------------------------
        switch = {
            0: 'BUTD',
            1: 'DFAF',
            2: 'MUAN',
            3: 'MCAN',
            4: 'SelRes',
            5: 'MEDAN',
            6: 'MCAoAN',
            7: 'MCANed',
            8: 'DPCM',
            9: 'DUIM',
        }

        if __C.BACKBONE == 0:
            self.backbone = DCN(__C)
        elif __C.BACKBONE == 1:
            self.backbone = DFAF(__C)
        elif __C.BACKBONE == 2:
            self.backbone = MUAN(__C)
        elif __C.BACKBONE == 3:
            self.backbone = MCAN(__C)
        elif __C.BACKBONE == 4:
            self.backbone = SelRes(__C)
        elif __C.BACKBONE == 5:
            self.backbone = MEDAN(__C)
        elif __C.BACKBONE == 6:
            self.backbone = MCAoAN(__C)
        elif __C.BACKBONE == 7:
            self.backbone = MCA_ED(__C)
        elif __C.BACKBONE == 8:
            self.backbone = AIN(__C)
        elif __C.BACKBONE == 9:
            self.backbone = FUIN(__C)
        else:
            print("backbone wrong!")
            exit()
        print('== backbone: ' + switch.get(__C.BACKBONE))


        self.attflat_img = AttFlat(__C)
        self.attflat_lang = AttFlat(__C)

        self.proj_norm = LayerNorm(__C.FLAT_OUT_SIZE)
        self.proj = nn.Linear(__C.FLAT_OUT_SIZE, answer_size)


    def forward(self, img_feat, ques_ix):

        # Make mask
        lang_feat_mask = self.make_mask(ques_ix.unsqueeze(2))
        img_feat_mask = self.make_mask(img_feat)

        # Pre-process Language Feature
        lang_feat = self.embedding(ques_ix)
        lang_feat, _ = self.lstm(lang_feat)

        # Pre-process Image Feature
        img_feat = self.img_feat_linear(img_feat)

        # Backbone Framework
        lang_feat, img_feat = self.backbone(
            lang_feat,
            img_feat,
            lang_feat_mask,
            img_feat_mask
        )

        lang_feat = self.attflat_lang(
            lang_feat,
            lang_feat_mask
        )

        img_feat = self.attflat_img(
            img_feat,
            img_feat_mask
        )

        proj_feat = lang_feat + img_feat
        proj_feat = self.proj_norm(proj_feat)
        proj_feat = torch.sigmoid(self.proj(proj_feat))

        return proj_feat


    # Masking
    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)
