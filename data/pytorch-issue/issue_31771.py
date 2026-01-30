import random

import torch
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

np.random.seed(5)

class RandomDataset(Dataset):
    def __init__(self):
        self.data = torch.rand(22, 1, 32, 32)
        self.name = torch.arange(1, 22)
        
    def __getitem__(self, idx):
        return self.name[idx], self.data[idx]
    
    def __len__(self):
        return len(self.data)
    

torch.distributed.init_process_group(backend="nccl")

dataset = RandomDataset()
sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True) 
dataloader = DataLoader(dataset, 
    batch_size=5,   
    pin_memory=True, drop_last=True, sampler=sampler) 

for epoch in range(3):
    print("epoch: ", epoch)
    for i, data in enumerate(dataloader, 0):
        names, _ = data
        print(names)

g = torch.Generator ()
g.manual_seed (self.epoch)

g = torch.Generator ()
g.manual_seed (self.epoch) 
self.g = g