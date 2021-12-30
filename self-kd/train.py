from __future__ import print_function, division, absolute_import
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable, grad
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import models
from autoaugment import CIFAR10Policy
from cutout import Cutout
from losses import DistillCE, FeatureLoss, DistillKL, SelfAttentionLoss, FTLoss
from models.util import SelfAttention


import argparse
import os, sys
import time
import logging

from utils import *

parser = argparse.ArgumentParser(description='Self-Distillation CIFAR Training')
parser.add_argument('--seed', type=int, default=2, help='random seed')

# path
parser.add_argument('--dataroot', type=str, default='../data')
parser.add_argument('--saveroot', type=str, default='./results', help='models and logs are saved here')

# basic arguments
parser.add_argument('--model', default="CIFAR_ResNet18", type=str)
parser.add_argument('--dataset', default='cifar100', type=str, help="cifar100|cifar10")

parser.add_argument('--autoaugment', default=True, type=bool)

# training hyperparameter
parser.add_argument('--name', default='0', type=str, help='name of run')
parser.add_argument('--epoch', default=250, type=int, help="epoch for training")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--batch_size', type=int, default=128, help="the sizeof batch")
parser.add_argument('--loss_coefficient', default=0.3, type=float)
parser.add_argument('--feature_loss_coefficient', default=0.03, type=float)
parser.add_argument('--print_freq', type=int, default=50, help='frequency of showing training results on console')
parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
parser.add_argument('--ngpu', default=1, type=int, help='number of gpu')
parser.add_argument('--sgpu', default=0, type=int, help='gpu index (start)')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--lamda', default=1.0, type=float, help='cls loss weight ratio')
parser.add_argument('--decay', type=float, default=5e-4, help='weight decay')
# distill arguments
parser.add_argument('--temp', '-T', type=float, default=3.0)

args = parser.parse_args()
use_cuda = torch.cuda.is_available()

best_val = 0  # best validation accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

cudnn.benchmark = True

args.model_name = '{}_{}_aug_{}'.format(args.model, args.dataset, args.autoaugment)

# Data
print('==> Preparing dataset: {}'.format(args.dataset))
    
if args.autoaugment:
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4, fill=128),
                             transforms.RandomHorizontalFlip(), CIFAR10Policy(), transforms.ToTensor(),
                             Cutout(n_holes=1, length=16),
                             transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))]) 
else:
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4, fill=128),
                                          transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                          transforms.Normalize((0.5071, 0.4865, 0.4409),
                                                               (0.2673, 0.2564, 0.2762))])
    
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
])

if args.dataset == "cifar100":
    trainset = torchvision.datasets.CIFAR100(
        root=args.data_root,
        train=True,
        download=True,
        transform=transform_train
    )
    testset = torchvision.datasets.CIFAR100(
        root=args.data_root,
        train=False,
        download=True,
        transform=transform_test
    )
    
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=4
)
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=4
)

num_class = trainloader.dataset.num_classes
print('Number of train dataset: ' ,len(trainloader.dataset))
print('Number of validation dataset: ' ,len(testloader.dataset))

# Model
print('==> Building model: {}'.format(args.model))
model = models.load_model(args.model, num_class)

logdir = os.path.join(args.saveroot, args.dataset, args.model, args.name+'_'+args.distill, 'temp_'+ str(args.temp)+'_lamda_'+str(args.lamda)+'_beta_'+str(args.beta))
set_logging_defaults(logdir, args)
logger = logging.getLogger('main')
logname = os.path.join(logdir, 'log.csv')

data = torch.rand(2, 3, 32, 32)
model.eval()
_, feat_s = model(data, is_feat=True)

module_list = nn.ModuleList([])
module_list.append(model)
trainable_list = nn.ModuleList([])
trainable_list.append(model)

criterion_cls = nn.CrossEntropyLoss()

s_n = [f.shape[1] for f in feat_s[1:]]
t_n = feat_s[0]

criterion_kd = SelfAttentionLoss(args)
self_attention = SelfAttention(len(feat_s)-1, 1, args.batch_size, s_n, t_n)
module_list.append(self_attention)
trainable_list.append(self_attention)

criterion_list = nn.ModuleList([])
criterion_list.append(criterion_cls)    # classification loss
criterion_list.append(criterion_kd)     # other knowledge distillation loss

logger.info("args = %s", args)
    
logger.info('----------- Network Initialization --------------')
logger.info('%s', model)
logger.info('param_size = %fMB', count_parameters_in_MB(trainable_list))
logger.info('----------------------------')

if use_cuda:
    torch.cuda.set_device(args.sgpu)
    module_list.cuda()
    criterion_list.cuda()
    print(torch.cuda.device_count())
    print('Using CUDA..')

if args.ngpu > 1:
    net = torch.nn.DataParallel(module_list, device_ids=list(range(args.sgpu, args.sgpu + args.ngpu)))
    
optimizer = optim.SGD(trainable_list.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)

logdir = os.path.join(args.saveroot, args.dataset, args.model, args.name+'_'+args.distill, 'temp_'+ str(args.temp)+'_lamda_'+str(args.lamda)+'_beta_'+str(args.beta))
set_logging_defaults(logdir, args)
logger = logging.getLogger('main')

# Resume
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(os.path.join(logdir, 'ckpt.t7'))
    net.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1
    rng_state = checkpoint['rng_state']
    torch.set_rng_state(rng_state)

def main():
    for epoch in range(start_epoch, args.epoch):
        weight = train(epoch)
        val_loss, val_acc, val_accen = test(epoch, weight)
        adjust_learning_rate(optimizer, epoch)
        
    print("Best Accuracy : {}".format(best_val))
    logger = logging.getLogger('best')
    logger.info('[Acc {:.3f}]'.format(best_val))
    
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    correct_ensemble = 0
    total = 0
    train_kd_loss = 0
    
    end = time.time()
    for batch_idx, (inputs, labels) in enumerate(trainloader):
        if use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        outputs, outputs_feature = net(inputs)
            
        # compute loss
        criterion_cls = criterion_list[0]
        criterion_kd = criterion_list[1]
        
        #   for deepest classifier
        loss = criterion_cls(outputs[0], labels) 
        train_loss += loss.item()
        
        teacher_feature = outputs_feature[0].detach()
        s_value, f_target, weight = module_list[1](outputs_feature[1:], teacher_feature)
        loss_kd = criterion_kd(s_value, f_target, outputs, weight)
        
        loss += args.lamda * loss_kd
        train_kd_loss += loss_kd.item()
        
        _, predicted = torch.max(outputs[0], 1)
        _, predicted_ensemble = torch.max(outputs[0]/2.0+torch.mul(outputs[1:],weight/2.0), 1)
        
        total += labels.size(0)
        correct += predicted.eq(labels.data).sum().float().cpu()
        correct_ensemble += predicted_ensemble.eq(labels.data).sum().float().cpu()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        progress_bar(batch_idx, len(trainloader),
                     'Loss: %.3f | Acc: %.3f%% (%d/%d) | Acc_en: %.3f%% (%d/%d) | KD: %.3f '
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total, 100.*correct_ensemble/total, correct_ensemble, total, train_kd_loss/(batch_idx+1)))

        logger = logging.getLogger('train')
        logger.info('[Epoch {}] [Loss {:.3f}] [cls {:.3f}] [Acc {:.3f}] [Acc_en {:.3f}]'.format(
            epoch,train_loss/(batch_idx+1),
            train_kd_loss/(batch_idx+1),
            100.*correct/total,
            100.*correct_ensemble/total))
    
    return weight
            
def test(epoch, weight):
    global best_val
    net.eval()
    val_loss = 0.0
    correct = 0.0
    correct_ensemble = 0.0
    total = 0.0
    
    # Define a data loader for evaluating
    loader = testloader
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(loader):
            if use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            outputs, _ = net(inputs)
            criterion_cls = criterion_list[0]
            loss = torch.mean(criterion_cls(outputs[0], labels))

            val_loss += loss.item()
            _, predicted = torch.max(outputs[0], 1)
            _, predicted_ensemble = torch.max(outputs[0]/2.0+torch.mul(outputs[1:],weight/2.0), 1)
            
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum().float()
            correct_ensemble += predicted_ensemble.eq(labels.data).cpu().sum().float()

            progress_bar(batch_idx, len(loader),
                         'Loss: %.3f | Acc: %.3f%% (%d/%d) | Acc_en: %.3f%% (%d/%d)'
                         % (val_loss/(batch_idx+1), 100.*correct/total, correct, total, 100.*correct_ensemble/total, correct, total))

    acc = 100.*correct/total
    acc_en = 100.*correct_ensemble/total
    logger = logging.getLogger('val')
    logger.info('[Epoch {}] [Loss {:.3f}] [Acc {:.3f}] [Acc {:.3f}]'.format(
        epoch,
        val_loss/(batch_idx+1),
        acc, acc_en))
    
    if acc > best_val:
        best_val = acc_en
        checkpoint(acc_en, epoch)
    
    return (val_loss/(batch_idx+1), acc, acc_en)

def checkpoint(acc, epoch):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    torch.save(state, os.path.join(logdir, 'ckpt.t7'))

def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = args.lr
    if epoch >= args.epoch // 3:
        lr /= 10
    if epoch >= args.epoch*2 // 3:
        lr /= 10
    if epoch >= args.epoch - 10:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    main()

        
        
        
        
    
        
                
        
        
        
        
                
                
        
        
        
    
        
    



