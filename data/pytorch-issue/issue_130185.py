import torch

my_tensor = torch.tensor([])

torch.aminmax(input=my_tensor) # Error

import torch

my_tensor = torch.tensor([])

torch.aminmax(input=my_tensor, dim=0) # Error

import torch

my_tensor = torch.tensor([[]])

torch.aminmax(input=my_tensor) # Error

my_tensor = torch.tensor([[[]]])

torch.aminmax(input=my_tensor) # Error

import torch

my_tensor = torch.tensor([[]])

torch.aminmax(input=my_tensor, dim=0)
# torch.return_types.aminmax(
# min=tensor([]),
# max=tensor([]))

my_tensor = torch.tensor([[[]]])

torch.aminmax(input=my_tensor, dim=0)
# torch.return_types.aminmax(
# min=tensor([], size=(1, 0)),
# max=tensor([], size=(1, 0)))

import torch

torch.__version__ # 2.3.0+cu121