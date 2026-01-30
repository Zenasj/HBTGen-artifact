import torch.nn.functional as F
import torchvision

import torchdynamo
from torchdynamo import optimize
import os

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import time
import torchmetrics
import pytorch_lightning as pl
import kornia as K

import numpy as np
from PIL import Image


import functools
import time

def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        print(f"Elapsed time: {elapsed_time:0.4f} seconds")
        return value
    return wrapper_timer

"""## Define Data Augmentations module"""

class DataAugmentation(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""
    def __init__(self, apply_color_jitter: bool = False) -> None:
        super().__init__()
        self._apply_color_jitter = apply_color_jitter

        self._max_val: float = 255.

        self.transforms = nn.Sequential(
            K.enhance.Normalize(0., self._max_val),
            K.augmentation.RandomHorizontalFlip(p=0.5)
        ).to(torch.device("cuda:0"))

        self.jitter = K.augmentation.ColorJitter(0.5, 0.5, 0.5, 0.5)

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_out = self.transforms(x)  # BxCxHxW
        if self._apply_color_jitter:
            x_out = self.jitter(x_out)
        return x_out

"""## Define a Pre-processing model"""



class CoolSystem(nn.Module):

    def __init__(self):
        super(CoolSystem, self).__init__()
        self.l1 = torch.nn.Linear(3 * 32 * 32, 10)
        self.transform = DataAugmentation()


    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        # REQUIRED
        x,y = batch
        x_aug = self.transform(x)  # => we perform GPU/Batched data augmentation
        logits = self.forward(x_aug)
        loss = F.cross_entropy(logits, y)
        return loss


"""## Run training"""

# init model
model = CoolSystem().to(torch.device("cuda:0"))
x = torch.Tensor(1,3072).to(torch.device("cuda:0"))
y = torch.ones([1], dtype=torch.long, device=torch.device('cuda:0'))
batch = (x, y)

@timer
def normal_train():
    for i in range(1000):
        model.training_step(batch, i)

@timer
@torchdynamo.optimize("inductor")
def inductor_train():
    for i in range(1000):
        model.training_step(batch, i)

normal_train()
inductor_train()