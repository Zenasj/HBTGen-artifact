import torch.nn as nn

import torch
from torch import nn

torch.set_default_device(device='cuda')
torch.set_default_dtype(d=torch.float64)

tran = nn.Transformer().generate_square_subsequent_mask(sz=3)
tran.device, tran.dtype
# (device(type='cpu'), torch.float32) # Here

import torch

torch.set_default_device(device='cuda')
torch.set_default_dtype(d=torch.float64)

my_tensor = torch.tensor([0., 1., 2.])
my_tensor.device, my_tensor.dtype
# (device(type='cuda', index=0), torch.float64)

import torch

torch.__version__ # 2.4.1+cu121