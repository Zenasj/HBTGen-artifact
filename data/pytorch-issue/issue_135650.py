import torch.nn as nn

import torch
from torch import nn

my_tensor = torch.tensor([[0., 1., 2.]])

torch.manual_seed(42)        # ↓↓↓↓                            # ↓↓↓↓ 
conv1d = nn.Conv1d(in_channels=True, out_channels=3, kernel_size=True)
conv1d(input=my_tensor)
# tensor([[-0.7336, -1.2205, -1.7073],
#         [0.8692, 1.4565, 2.0438],
#         [0.1872, 1.0687, 1.9502]], grad_fn=<SqueezeBackward1>)

torch.manual_seed(42)        # ↓↓↓↓                             # ↓↓↓↓ 
conv1d = nn.Conv1d(in_channels=True, out_channels=3, kernel_size=(True,))
conv1d(input=my_tensor)
# tensor([[-0.7336, -1.2205, -1.7073],
#         [0.8692, 1.4565, 2.0438],
#         [0.1872, 1.0687, 1.9502]], grad_fn=<SqueezeBackward1>)

import torch
from torch import nn

my_tensor = torch.tensor([[0., 1., 2.]])

torch.manual_seed(42)                        # ↓↓↓↓                      
conv1d = nn.Conv1d(in_channels=1, out_channels=True, kernel_size=1)
conv1d(input=my_tensor)

import torch

torch.__version__ # 2.4.0+cu121