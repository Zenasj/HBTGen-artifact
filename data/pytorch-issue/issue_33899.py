import random

import torch
import numpy as np
device = torch.device('cuda')
a = torch.Tensor(np.random.normal(size=10)).to(device)
torch.Tensor(a)

import torch

a = torch.tensor(1, device='cuda')
b = torch.Tensor(a)
print(b)