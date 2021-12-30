from __future__ import print_function, division, absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillKL(nn.Module):
    def __init__(self):
        super(DistillKL, self).__init__()
       # self.temp_factor = temp_factor
        self.kl_div = nn.KLDivLoss(reduction="sum")

    def forward(self, input, target, temp):
        log_p = torch.log_softmax(input/temp, dim=1)
        q = torch.softmax(target/temp, dim=1)
        loss = self.kl_div(log_p, q)*(temp**2)/input.size(0)
        return loss

class DistillCE(nn.Module):
    def __init__(self, args):
        super(DistillCE, self).__init__()
        self.temperature = args.temperature
        
    def forward(self, outputs, targets):
        log_softmax_outputs = F.log_softmax(outputs /self.temperature, dim=1)
        softmax_targets = F.softmax(targets/self.temperature, dim=1)
        loss = -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()
        return loss