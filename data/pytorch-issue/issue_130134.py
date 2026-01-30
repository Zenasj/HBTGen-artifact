import torch

my_tensor = torch.tensor([])

torch.kthvalue(input=my_tensor, k=1) # Error

import torch

my_tensor = torch.tensor([])

torch.kthvalue(input=my_tensor, k=1, dim=0) # Error

import torch

my_tensor = torch.tensor([[]])

torch.kthvalue(input=my_tensor, k=1) # Error

my_tensor = torch.tensor([[[]]])

torch.kthvalue(input=my_tensor, k=1) # Error

import torch

my_tensor = torch.tensor([[]])

torch.kthvalue(input=my_tensor, k=1, dim=0)
# torch.return_types.kthvalue(
# values=tensor([]),
# indices=tensor([], dtype=torch.int64))

my_tensor = torch.tensor([[[]]])

torch.kthvalue(input=my_tensor, k=1, dim=0)
# torch.return_types.kthvalue(
# values=tensor([], size=(1, 0)),
# indices=tensor([], size=(1, 0), dtype=torch.int64))

import torch

torch.__version__ # 2.3.0+cu121