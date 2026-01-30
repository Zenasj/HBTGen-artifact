import torch

torch.set_default_device(device='cuda:0')
torch.get_default_device() # Error

import torch

torch.__version__ # 2.2.1+cu121

import torch

torch.set_default_device(device='cuda:0')
torch.get_default_device() # device(type='cuda', index=0)

torch.__version__ # 2.3.0