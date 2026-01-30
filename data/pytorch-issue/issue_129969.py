import torch

torch.full(size=(3,), fill_value=torch.tensor([8])) # Error

import torch

torch.full(size=(3,), fill_value=torch.tensor(8)) # tensor([8, 8, 8])

import torch

torch.__version__ # 2.3.0+cu121