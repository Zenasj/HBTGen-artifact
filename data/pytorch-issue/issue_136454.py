import torch
from torch import linalg

my_tensor = torch.tensor([[[8., -3., 0., 1.]]])
                           # ↓ ↓ ↓ ↓ ↓
linalg.norm(input=my_tensor, ord='fro', dim=(0, 1, 2)) # Error
                           # ↓ ↓ ↓ ↓ ↓
linalg.norm(input=my_tensor, ord='nuc', dim=(0, 1, 2)) # Error

import torch
from torch import linalg

my_tensor = torch.tensor([[[8., -3., 0., 1.]]])
                           # ↓ ↓ ↓ ↓
linalg.norm(input=my_tensor, ord=None, dim=(0, 1, 2)) # Error
                           # ↓ ↓ ↓
linalg.norm(input=my_tensor, ord=2, dim=(0, 1, 2)) # Error

import torch

torch.__version__ # 2.4.1+cu121