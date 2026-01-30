import torch.nn as nn

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import time


class ToyDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 1000

    def __getitem__(self, item):
        d1, d2, d3 = np.random.randint(50, 100, size=3)
        data = np.random.randn(d1, d2, d3).astype(np.float32)
        return data[None, ...]


class ToyModel(torch.nn.Module):
    def __init__(self, out_channels=10, num_convs=3):
        super(ToyModel, self).__init__()
        self.module_dict = torch.nn.ModuleDict()
        for i in range(num_convs):
            conv_op = torch.nn.Conv3d(in_channels=1 if i == 0 else out_channels,
                                      out_channels=out_channels,
                                      kernel_size=3)
            self.module_dict[f"conv_{i}"] = conv_op

    def forward(self, x):
        for k, op in self.module_dict.items():
            x = op(x)
        return x


def train(model, device, loader, n_epochs=10):
    time_init = time.time()
    for epoch in range(n_epochs):
        for input_tensor in loader:
            with torch.no_grad():
                input_tensor = torch.from_numpy(input_tensor[None, ...]).to(device)
                _ = model(input_tensor) # Commenting this removes the leak, data has to be used by the model
        print(epoch, time.time() - time_init)


train_loader = DataLoader(dataset=ToyDataset(), collate_fn=lambda x: x[0], num_workers=4)
device = f'cuda' if torch.cuda.is_available() else 'cpu'
model = ToyModel().to(device)
train(model=model, device=device, loader=train_loader, n_epochs=100)

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import time
import resource
import random

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

class ToyDataset(Dataset):
    def __init__(self):
        self.enable_variable_size = True

    def __len__(self):
        return 1000

    def __getitem__(self, item):
        if self.enable_variable_size:
          d1, d2, d3 = np.random.randint(50, 100, size=3)
        else:
          d1, d2, d3 = 100, 100, 100
        data = np.random.randn(d1, d2, d3).astype(np.float32)
        return data[None, ...]


class ToyModel(torch.nn.Module):
    def __init__(self, out_channels=10, num_convs=3):
        super(ToyModel, self).__init__()
        self.module_dict = torch.nn.ModuleDict()
        for i in range(num_convs):
            conv_op = torch.nn.Conv3d(in_channels=1 if i == 0 else out_channels,
                                      out_channels=out_channels,
                                      kernel_size=3)
            self.module_dict[f"conv_{i}"] = conv_op

    def forward(self, x):
        for k, op in self.module_dict.items():
            x = op(x)
        return x


def train(model, device, loader, n_epochs=10):
    print("starting training..")
    time_init = time.time()
    for epoch in range(n_epochs):
        for input_tensor in loader:
            with torch.no_grad():
                input_tensor = torch.from_numpy(input_tensor[None, ...]).to(device)
                _ = model(input_tensor)
        print(f"epoch: [{epoch}] - max cpu memory: {get_cpu_memory()}MiB")

def get_cpu_memory():
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 10 ** 3 / 2 ** 20)


train_loader = DataLoader(dataset=ToyDataset(), collate_fn=lambda x: x[0], num_workers=2)
device = f'cuda' if torch.cuda.is_available() else 'cpu'
model = ToyModel().to(device)
train(model=model, device=device, loader=train_loader, n_epochs=100)

import random
import torch
from torch import nn
import torch.utils.checkpoint as cp
from tqdm import tqdm
class FullyConvNet(nn.Module):
    def __init__(self):
        super(FullyConvNet, self).__init__()

        # 卷积层
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)

        # 全卷积层
        self.fc1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.fc2 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        self.fc3 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)

    def forward(self, x):
        # 卷积层
        x = self.conv1(x)
        x = nn.functional.relu(x, inplace=True)
        x = nn.functional.max_pool2d(x, 2)

        x = self.conv2(x)
        x = nn.functional.relu(x, inplace=True)
        x = nn.functional.max_pool2d(x, 2)

        x = self.conv3( x)
        x = nn.functional.relu(x, inplace=True)
        x = nn.functional.max_pool2d(x, 2)

        # 全卷积层
        x = self.fc1(x)
        x = nn.functional.relu(x, inplace=True)
        x = self.fc2(x)
        x = nn.functional.relu(x, inplace=True)

        x = self.fc3(x)

        # 返回结果
        return x

model = FullyConvNet()
model = model.to("cuda:0")

for epoch in tqdm(range(100)):
    for i in tqdm(range(100000)):
        images = torch.randn((8, 3, random.randint(32,448),random.randint(224,1024)))
        images = images.to("cuda:0")
        outputs = model(images)

import random
import torch
from torch import nn
import torch.utils.checkpoint as cp

import resource

class FullyConvNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        self.fc1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.fc2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.fc3 = nn.Conv2d(in_channels=256, out_channels=2, kernel_size=3, padding=1)

    def forward(self, x):
        x = cp.checkpoint(self.conv1, x)
        x = nn.functional.relu(x,inplace=True)
        x = nn.functional.max_pool2d(x, 2)

        x = cp.checkpoint(self.conv2, x)
        x = nn.functional.relu(x,inplace=True)
        x = nn.functional.max_pool2d(x, 2)

        x = cp.checkpoint(self.conv3, x)
        x = nn.functional.relu(x,inplace=True)
        x = nn.functional.max_pool2d(x, 2)

        x = cp.checkpoint(self.fc1, x)
        x = nn.functional.relu(x,inplace=True)
        x = cp.checkpoint(self.fc2, x)
        x = nn.functional.relu(x,inplace=True)
        # x = self.fc3(x)

        return x

model = FullyConvNet()
model = model.to("cuda:0")

for i in range(1000):
    images = torch.randn((8, 3, random.randint(224, 448), random.randint(224, 448)), device='cuda:0')
    model(images)
    max_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # MB
    max_cuda_mem = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
    print(f"iter: {i} cpu: {max_mem} cuda: {max_cuda_mem}")