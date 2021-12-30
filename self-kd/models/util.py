from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class SelfAttention(nn.Module):
    def __init__(self, s_len, input_channel, s_n, s_t, factor=4): 
        super(SelfAttention, self).__init__()
          
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        setattr(self, 'query_weight_teacher', MLPEmbed(input_channel, input_channel//factor))
        for i in range(s_len):
            setattr(self, 'key_weight'+str(i), MLPEmbed(input_channel, input_channel//factor))
        
        for i in range(s_len):
            setattr(self, 'regressor'+str(i), AAEmbed(s_n[i], s_t))
                
    def forward(self, feat_s, feat_t):
        
        sim_s = list(range(len(feat_s)))
        bsz = feat_s[0].shape[0]
        # similarity matrix
        sim_temp = feat_t.reshape(bsz, -1)
        sim_t = torch.matmul(sim_temp, sim_temp.t())
        for i in range(len(feat_s)):
            sim_temp = feat_s[i].reshape(bsz, -1)
            sim_s[i] = torch.matmul(sim_temp, sim_temp.t())
        
        # query of target layers    
        proj_query = self.query_weight_teacher(sim_t)
        proj_query = proj_query[:, None, :]
        
        # key of source layers   
        proj_key = self.key_weight0(sim_s[0])
        proj_key = proj_key[:, :, None]
        for i in range(1, len(sim_s)):
            temp_proj_key = getattr(self, 'key_weight'+str(i))(sim_s[i])
            proj_key = torch.cat([proj_key, temp_proj_key[:, None, :]], 1)
            
        # attention weight
        energy = torch.bmm(proj_query, proj_key) # batch_size X No.stu feature X No.tea feature
        attention = F.softmax(energy, dim = -1)
        attention = attention.view(attention.size(0), -1)
        
        # feature space alignment
        proj_value_stu = []
        for i in range(len(sim_s)):
            #proj_value_stu.append          
            s_H, t_H = feat_s[i].shape[2], feat_t.shape[2]
            if s_H > t_H:
                input = F.adaptive_avg_pool2d(feat_s[i], (t_H, t_H))
                proj_value_stu[i] = getattr(self, 'regressor'+str(i))(input)
                value_tea = feat_t
            elif s_H < t_H or s_H == t_H:
                target = F.adaptive_avg_pool2d(feat_t, (s_H, s_H))
                proj_value_stu[i] = getattr(self, 'regressor'+str(i))(feat_s[i])
                value_tea = target
                
        return proj_value_stu, value_tea, attention
        
           
            

class AAEmbed(nn.Module):
    """non-linear embed by MLP"""
    def __init__(self, num_input_channels=1024, num_target_channels=128):
        super(AAEmbed, self).__init__()
        self.num_mid_channel = 2 * num_target_channels
        
        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False)
        def conv3x3(in_channels, out_channels, stride=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        
        self.regressor = nn.Sequential(
            conv1x1(num_input_channels, self.num_mid_channel),
            nn.BatchNorm2d(self.num_mid_channel),
            nn.ReLU(inplace=True),
            conv3x3(self.num_mid_channel, self.num_mid_channel),
            nn.BatchNorm2d(self.num_mid_channel),
            nn.ReLU(inplace=True),
            conv1x1(self.num_mid_channel, num_target_channels),
        )

    def forward(self, x):
        x = self.regressor(x)
        return x
    
class MLPEmbed(nn.Module):
    """non-linear embed by MLP"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(MLPEmbed, self).__init__()
        self.linear1 = nn.Linear(dim_in, 2 * dim_out)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(2 * dim_out, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.relu(self.linear1(x))
        x = self.l2norm(self.linear2(x))
        return x


class Normalize(nn.Module):
    """normalization layer"""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out