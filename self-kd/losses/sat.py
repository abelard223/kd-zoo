from __future__ import print_function, division, absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fl import FTLoss
from .kd import DistillKL

class SelfAttentionLoss(nn.Module):
    def __init__(self, args):
        super(SelfAttentionLoss, self).__init__()
        self.criterion_kd = DistillKL()
        self.criterion_fl = FTLoss()
        self.loss_efficient = args.loss_efficient
        self.feature_loss_coefficient = args.feature_loss_coefficient
        
    def forward(self, s_value, f_target, outputs, weight):
        bsz, num_stu = weight.shape
        ind_loss_kd = torch.zeros(bsz, num_stu).cuda()
        ind_loss_fl = torch.zeros(bsz, num_stu).cuda()
        
        s_outputs = outputs[1:]
        t_outputs = outputs[0]
        
        for i in range(num_stu):
            ind_loss_fl[:, i] = self.criterion_fl(s_value[i], f_target).reshape(bsz,-1).mean(-1)
            ind_loss_kd[:, i] = self.criterion_kd(s_outputs[i], t_outputs).reshape(bsz,-1).mean(-1)

        loss_kd = (weight * ind_loss_kd).sum()/(1.0*bsz)
        loss_fl = (weight * ind_loss_fl).sum()/(1.0*bsz)
        return self.loss_efficient * loss_kd + self.feature_loss_coefficient * loss_fl
        