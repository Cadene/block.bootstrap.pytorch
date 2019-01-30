import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .fusions.factory import factory as factory_fusion
from .mlp import MLP

class VRDNet(nn.Module):

    def __init__(self, opt):
        super(VRDNet, self).__init__()
        self.opt = opt
        self.classeme_embedding = nn.Embedding(
            self.opt['nb_classeme'],
            self.opt['classeme_dim'])
        self.fusion_c = factory_fusion(self.opt['classeme'])
        self.fusion_s = factory_fusion(self.opt['spatial'])
        self.fusion_f = factory_fusion(self.opt['feature'])
        self.predictor = MLP(**self.opt['predictor'])

    def forward(self, batch):
        bsize = batch['subject_boxes'].size(0)
        x_c = [self.classeme_embedding(batch['subject_cls_id']),
               self.classeme_embedding(batch['object_cls_id'])]
        x_s = [batch['subject_boxes'], batch['object_boxes']]
        x_f = [batch['subject_features'], batch['object_features']]

        x_c = self.fusion_c(x_c)
        x_s = self.fusion_s(x_s)
        x_f = self.fusion_f(x_f)

        x = torch.cat([x_c, x_s, x_f], -1)

        if 'aggreg_dropout' in self.opt:
            x = F.dropout(x, self.opt['aggreg_dropout'], training=self.training)
        y = self.predictor(x)
        
        out = {
            'rel_scores': y
        }
        return out
