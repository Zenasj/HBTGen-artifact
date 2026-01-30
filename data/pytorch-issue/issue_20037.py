import random

import argparse                                                                                                                                                                                                                                     
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
from torch.utils import data
import torch.optim as optim
 
batch_size = 8
num_gpus = int(os.environ['WORLD_SIZE'])
 
class ToyDataset(data.Dataset):
    def __init__(self):
        super(ToyDataset, self).__init__()
 
    def __len__(self):
        return 3840
 
    def __getitem__(self, index):
        return np.random.rand(3, 800, 800).astype('f'), np.random.randint(0, 20, size=(800, 800)).astype('f')
 
 
class ToyNet(torch.nn.Module):
    def __init__(self):
        super(ToyNet, self).__init__()
        self.conv = torch.nn.Conv2d(3, 32, 1)
        self.final_conv = torch.nn.Conv2d(32, 20, 1)
 
    def forward(self, x):
        x = self.conv(x)
        x = self.final_conv(x)
        return x
 
 
def Main():
    print('Pytorch version: {}'.format(torch.__version__))
 
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()
 
    torch.distributed.init_process_group(backend='nccl')
 
    model = ToyNet()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
 
    device = torch.device('cuda', args.local_rank)
    model = model.to(device)
    distrib_model = torch.nn.parallel.DistributedDataParallel(model,
                                                              device_ids=[args.local_rank],
                                                              output_device=args.local_rank)
     
    train_set = ToyDataset()
     
    sampler = DistributedSampler(train_set)
    dataloader = data.DataLoader(
                     dataset=train_set,
                     batch_size=batch_size // num_gpus,
                     sampler=sampler)
     
    start_time = time.time()
    distrib_model.train()
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        predictions = distrib_model(inputs.to(device))
        loss = loss_function(predictions, labels.long().to(device))
        loss.backward()
        optimizer.step()
        #print('batch_idx: {}'.format(batch_idx))
    end_time = time.time()
    print('Total time: {}'.format(end_time-start_time))
     
     
if __name__ == "__main__":
    Main()