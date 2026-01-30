import torch.nn as nn

import torch
from torch import nn

my_tensor = torch.tensor([2., 7., 4.])
                             # ↓↓
linear = nn.Linear(in_features=3., out_features=5)

import torch
from torch import nn

my_tensor = torch.tensor([2., 7., 4.])
                                             # ↓↓↓↓↓↓
linear = nn.Linear(in_features=3, out_features=5.+0.j)

import torch

torch.__version__ # 2.4.0+cu121