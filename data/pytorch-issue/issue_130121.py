import torch

my_tensor = torch.tensor([])

torch.argmax(input=my_tensor) # Error

import torch

my_tensor = torch.tensor([])

torch.argmax(input=my_tensor, dim=0) # Error

import torch

my_tensor = torch.tensor([[]])

torch.argmax(input=my_tensor) # Error

my_tensor = torch.tensor([[[]]])

torch.argmax(input=my_tensor) # Error

import torch

my_tensor = torch.tensor([[]])

torch.argmax(input=my_tensor, dim=0)
# tensor([], dtype=torch.int64)

my_tensor = torch.tensor([[[]]])

torch.argmax(input=my_tensor, dim=0)
# tensor([], size=(1, 0), dtype=torch.int64)

import torch

torch.__version__ # 2.3.0+cu121