import torch.nn as nn

import random
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import psutil
import os

class ExampleDataset(Dataset):
    def __init__(self):
        self.num_samples = 1000

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        h = random.randint(666, 999)
        w = random.randint(666, 999)
        tensor = torch.ones((3, h, w))
        return tensor
    
dataset = ExampleDataset()
data_loader = DataLoader(dataset, batch_size=1, num_workers=0)
model = getattr(torchvision.models, 'resnet18')().to('cuda')

for epoch in range(10):
    for tensor in data_loader:
        tensor = tensor.to('cuda')
        output = model(tensor)
        print('memory used:', psutil.Process(os.getpid()).memory_info().rss / 1024.0 / 1024.0)

import random
import torch
import psutil
import os

model = torch.nn.Conv2d(in_channels=3, out_channels=2000, kernel_size=3, stride=1, padding=1).cuda()
leak_memory = True
h = w = 224

for epoch in range(100000):
    if leak_memory:
        h = random.randint(5, 800)
        w = random.randint(5, 800)
    tensor = torch.ones((1, 3, h, w)).cuda()
    output = model(tensor)
    print('\rmemory used:', psutil.Process(os.getpid()).memory_info().rss / 1024.0 / 1024.0, end='')