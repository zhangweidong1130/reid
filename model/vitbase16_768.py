import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .vision_transformer import vit_base

def make_model(args):
    return VITBASE(args)

class VITBASE(nn.Module):
    def __init__(self, args):
        super(VITBASE, self).__init__()
        self.args = args
        self.backbone = vit_base()


    def forward(self, y):
        x, dtype, labeld = y
        x = self.backbone(x)
        y, fc = self.fc(x)

        outputs = {'predict': y,
                   'global_feat':[y],
                   'fc_logits':[fc]
        }
        return outputs