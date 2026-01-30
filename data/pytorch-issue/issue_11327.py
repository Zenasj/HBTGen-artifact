import argparse,os,time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed
import torchvision
from torchvision import datasets, transforms
import numpy as np
import models
from util import *

parser = argparse.ArgumentParser()
parser.add_argument('--start_epoch', type=int, default=1, help='start epoch number')
parser.add_argument('--epoch', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate, default=0.1')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum, default=0.9')
parser.add_argument('--weight_decay', type=float, default=0.0002, help='weight_decay, default=0.0002')
parser.add_argument('--batch_s', type=int, default=64, help='input batch size')
parser.add_argument('--grid_s', type=int, default=8, help='grid size')
parser.add_argument('--data', type=str, default='../vgg2data', help='data directory')
parser.add_argument('--workers', type=int, default=1, help='number of data loading workers')
parser.add_argument('--output_dir', type=str, default='./output/', help='model_saving directory')
parser.add_argument('--resume', type=str, default='', help='resume')
parser.add_argument("--display_interval", type=int, default=50)
parser.add_argument("--local_rank", type=int)
opt = parser.parse_args()
torch.cuda.set_device(opt.local_rank)

dist.init_process_group(backend='nccl', init_method='env://', world_size=8)

train_dir = os.path.join(opt.data, 'train')
train_dataset = datasets.ImageFolder(
    train_dir,
    transforms.Compose([transforms.ToTensor()])
)
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_s,
        num_workers=opt.workers,
        pin_memory=True,
        shuffle=False,
        sampler=train_sampler
    )

input_size = (opt.batch_s, 3, 128, 128)
num_classes = 9092
#num_classes = 34
model = models.se_resnet34_v3(input_size, opt.grid_s, num_classes)
if opt.resume:
    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
            .format(opt.resume, checkpoint['epoch']))
model.cuda()
model = torch.nn.parallel.DistributedDataParallel(model,\
    device_ids=[opt.local_rank], output_device=opt.local_rank)

optimizer = optim.SGD([
        {'params': get_parameters(model, bias=False)},
        {'params': get_parameters(model, bias=True), 'lr':opt.lr * 2, 'weight_decay': 0},
        {'params': get_parameters(model, bn=True), 'lr':opt.lr * 1.00001001358, 'weight_decay':0}
    ], lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
if opt.resume:
    if os.path.isfile(opt.resume):
        checkpoint = torch.load(opt.resume)
        optimizer.load_state_dict(checkpoint['optimizer'])

scheduler = optim.lr_scheduler.MultiStepLR(optimizer, \
    milestones=[8,15,18], gamma=0.5)

def train(epoch):
    for batch_idx, (data, label) in enumerate(train_loader, 0):
        optimizer.zero_grad()
        data, label = data.cuda(), label.cuda()
        output, grid = model(data)
        nll_loss = F.nll_loss(output, label)
        de_loss = deformation_constraint_loss(grid, opt.grid_s)
        loss =  nll_loss + de_loss
        loss.backward()
        optimizer.step()
for epoch in range(opt.start_epoch, opt.epoch + 1):
    train_sampler.set_epoch(epoch)
    scheduler.step() 
    model.train()
    train(epoch)

import torch.multiprocessing as mp

if __name__ == '__main__':
      mp.set_start_method('forkserver')

import argparse,os,time
import torch.multiprocessing as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.utils.data.distributed
import torchvision
from torchvision import datasets, transforms
import numpy as np
import models
from utils import *
def train(epoch):
    for batch_idx, (data, label) in enumerate(train_loader, 0):
        optimizer.zero_grad()
        data, label = data.cuda(), label.cuda()

        output, grid = model(data)
        nll_loss = F.nll_loss(output, label)
        de_loss = deformation_constraint_loss(grid, opt.grid_s)
        loss =  nll_loss + de_loss
        loss.backward()
        optimizer.step()

if __name__ == '__main__':
    mp.set_start_method('forkserver')
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_epoch', type=int, default=1, help='start epoch number')
    parser.add_argument('--epoch', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate, default=0.1')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum, default=0.9')
    parser.add_argument('--weight_decay', type=float, default=0.0002, help='weight_decay, default=0.0002')
    parser.add_argument('--batch_s', type=int, default=64, help='input batch size')
    parser.add_argument('--grid_s', type=int, default=8, help='grid size')
    parser.add_argument('--data', type=str, default='../vgg2data', help='data directory')
    parser.add_argument('--workers', type=int, default=1, help='number of data loading workers')
    parser.add_argument('--output_dir', type=str, default='./output/', help='model_saving directory')
    parser.add_argument('--resume', type=str, default='', help='resume')
    parser.add_argument("--display_interval", type=int, default=50)
    parser.add_argument("--local_rank", type=int)
    opt = parser.parse_args()
    torch.cuda.set_device(opt.local_rank)
    
    dist.init_process_group(backend='nccl', init_method='env://', world_size=8)

    train_dir = os.path.join(opt.data, 'train')
    train_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose([transforms.ToTensor()])
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=opt.batch_s,
            num_workers=opt.workers,
            drop_last=True,
            pin_memory=False,
            shuffle=False,
            sampler=train_sampler
        )

    input_size = (opt.batch_s, 3, 128, 128)
    num_classes = 9092
    #num_classes = 34
    model = models.se_resnet34_v3(input_size, opt.grid_s, num_classes)
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                .format(opt.resume, checkpoint['epoch']))
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model,\
        device_ids=[opt.local_rank], output_device=opt.local_rank)

    optimizer = optim.SGD([
            {'params': get_parameters(model, bias=False)},
            {'params': get_parameters(model, bias=True), 'lr':opt.lr * 2, 'weight_decay': 0},
            {'params': get_parameters(model, bn=True), 'lr':opt.lr * 1.00001001358, 'weight_decay':0}
        ], lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)

    if opt.resume:
        if os.path.isfile(opt.resume):
            checkpoint = torch.load(opt.resume)
            optimizer.load_state_dict(checkpoint['optimizer'])

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, \
        milestones=[8,10,12,14,15,16,17,18,19,20,21,22,23,24], gamma=0.5)

    for epoch in range(opt.start_epoch, opt.epoch + 1):
        train_sampler.set_epoch(epoch)
        scheduler.step() 
        model.train()
        train(epoch)

import argparse,os,time
import torch.multiprocessing as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.utils.data.distributed
import torchvision
from torchvision import datasets, transforms
import numpy as np

class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def name(self):
        return "MLP"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_epoch', type=int, default=1, help='start epoch number')
    parser.add_argument('--epoch', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate, default=0.1')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum, default=0.9')
    parser.add_argument('--weight_decay', type=float, default=0.0002, help='weight_decay, default=0.0002')
    parser.add_argument('--batch_s', type=int, default=64, help='input batch size')
    parser.add_argument('--grid_s', type=int, default=8, help='grid size')
    parser.add_argument('--data', type=str, default='../vgg2data', help='data directory')
    parser.add_argument('--workers', type=int, default=1, help='number of data loading workers')
    parser.add_argument('--output_dir', type=str, default='./output/', help='model_saving directory')
    parser.add_argument('--resume', type=str, default='', help='resume')
    parser.add_argument("--display_interval", type=int, default=50)
    parser.add_argument("--local_rank", type=int)
    opt = parser.parse_args()

    torch.cuda.set_device(opt.local_rank)
    
    dist.init_process_group(backend='nccl', init_method='env://')

    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    mnist = torchvision.datasets.MNIST(root="~/Documents/workspace/datasets/mnist", train=True, transform=trans, download=True)

    train_sampler = torch.utils.data.distributed.DistributedSampler(mnist)
    train_loader = torch.utils.data.DataLoader(
            mnist,
            batch_size=opt.batch_s,
            num_workers=opt.workers,
            drop_last=True,
            pin_memory=False,
            shuffle=False,
            sampler=train_sampler
        )

    model = MLPNet().cuda()
    criterion = nn.CrossEntropyLoss()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.local_rank], output_device=opt.local_rank)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
        milestones=[8,10,12,14,15,16,17,18,19,20,21,22,23,24], gamma=0.5)
    
    for epoch in range(opt.start_epoch, opt.epoch + 1):
        print(epoch)
        train_sampler.set_epoch(epoch)
        scheduler.step()
        for batch_idx, (x, target) in enumerate(train_loader):
          optimizer.zero_grad()
          x, target = x.cuda(), target.cuda()
          out = model(x)
          loss = criterion(out, target)
          loss.backward()
          optimizer.step()