import torch

my_tensor = torch.tensor([])

torch.amin(input=my_tensor) # Error

import torch

my_tensor = torch.tensor([])

torch.amin(input=my_tensor, dim=0) # Error

import torch

my_tensor = torch.tensor([[]])

torch.amin(input=my_tensor) # Error

my_tensor = torch.tensor([[[]]])

torch.amin(input=my_tensor) # Error

import torch

my_tensor = torch.tensor([[]])

torch.amin(input=my_tensor, dim=0)
# tensor([])

my_tensor = torch.tensor([[[]]])

torch.amin(input=my_tensor, dim=0)
# tensor([], size=(1, 0))

import torch

torch.__version__ # 2.3.0+cu121