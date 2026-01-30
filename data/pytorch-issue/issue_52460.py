import torch.nn as nn

import torch
from torch import nn, einsum
import torch.nn.functional as F
import os
import numpy as np
import random


def seed_torch(seed=1029):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True

seed_torch()

class MyLayer(nn.Module):
    def __init__(self, dim, heads, dim_heads = 32):
        super().__init__()
        self.hidden_dim = heads * dim_heads
        self.to_qkv = nn.Conv1d(dim, self.hidden_dim * 3, 1, bias = False)

    def forward(self, x):
        x = torch.cat((x.mean(dim=-1), x.mean(dim=-2)), dim=-1)
        return self.to_qkv(x)

x = torch.randn(1, 32, 256, 512)

attn = MyLayer(
    dim = 32,
    heads = 8,
    dim_heads = 64
)
x1 = attn(x)

attn_c = MyLayer(
    dim = 32,
    heads = 8,
    dim_heads = 64
)
x2 = attn_c(x)