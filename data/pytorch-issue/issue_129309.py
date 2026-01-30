import torch

my_tensor = torch.tensor([7.+5.j])

torch.argsort(input=my_tensor) # Error

my_tensor = torch.tensor([[7.+5.j]])

torch.argsort(input=my_tensor) # Error

import torch

my_tensor = torch.tensor(7.+5.j)

torch.argsort(input=my_tensor)
# tensor(0)

import torch

torch.__version__ # 2.3.0+cu121