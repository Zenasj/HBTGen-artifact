import os
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data.distributed import DistributedSampler

parser = argparse.ArgumentParser(description='PyTorch ImagNet Example')

parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:23456', type=str,
                    help='url used to set up distributed training')


def imagenet_loader(train_batch_size):
    train_set = datasets.ImageFolder('/path/to/imagenet/',
                        transform = transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                          std=[0.229, 0.224, 0.225]), ]))
    train_sampler = DistributedSampler(train_set)
    train_loader = DataLoader(
        train_set, batch_size = train_batch_size, shuffle=False, sampler=train_sampler,
        num_workers = 4, pin_memory = False, drop_last=True)

    return train_sampler, train_loader

def main():
    args = parser.parse_args()

    ngpus_per_node = torch.cuda.device_count()
    print('GPU nums: ', ngpus_per_node, '\n')

    args.world_size = ngpus_per_node * args.world_size
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    print(args.gpu, torch.get_num_threads())

    args.rank = args.rank * ngpus_per_node + gpu
    dist.init_process_group(backend="nccl", init_method=args.dist_url, 
                       world_size=args.world_size, rank=args.rank)

    cudnn.benchmark = True
    #model = resnet18()

    train_sampler, train_loader = imagenet_loader(train_batch_size=64)

    for epoch in range(1, 1 + 1):
        train_sampler.set_epoch(epoch)
        train(train_loader, args)

def train(train_loader, args):
    if args.gpu == 0:
        t0 = time.time()
    else:
        t1 = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        
        if batch_idx % 100 == 0:
            if args.gpu == 0:
                end_t0 = time.time()
                print(args.gpu, batch_idx, data.size(0), '  Batch time:  ', end_t0-t0)
                t0 = end_t0
            else:
                end_t1 = time.time()
                print(args.gpu, batch_idx, data.size(0), '  Batch time:  ', end_t1-t1)
                t1 = end_t1

if __name__ == "__main__":
    main()

import os
import time
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def imagenet_loader(train_batch_size):
    train_set = datasets.ImageFolder('/path/to/imagenet/',
                        transform = transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                               std=[0.229, 0.224, 0.225]),  ]))
    train_loader = DataLoader(
        train_set, batch_size = train_batch_size, shuffle=True, 
        num_workers = 4, pin_memory = False, drop_last=True)

    return train_loader

if __name__ == "__main__":

    train_loader = imagenet_loader(train_batch_size=128)

    t0 = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        
        if batch_idx % 100 == 0:
            end_t0 = time.time()
            print(batch_idx, data.size(0), '  Batch time:  ', end_t0-t0)
            t0 = end_t0

# data = data.cuda(args.gpu, non_blocking=True)
       # target = target.cuda(args.gpu, non_blocking=True)

self.model.module.load_state_dict(torch.load(save_path + '/model_{}'.format(fixed_str)))
if not model_only:
    self.head.module.load_state_dict(torch.load(save_path + '/head_{}'.format(fixed_str)))
    self.head2.module.load_state_dict(torch.load(save_path + '/head2_{}'.format(fixed_str)))
    #self.optimizer.load_state_dict(torch.load(save_path + '/optimizer_{}'.format(fixed_str)))