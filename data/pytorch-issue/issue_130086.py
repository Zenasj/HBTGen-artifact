import torch

my_tensor = torch.tensor([])

torch.max(input=my_tensor) # Error

import torch

my_tensor = torch.tensor([])

torch.max(input=my_tensor, dim=0) # Error

import torch

my_tensor = torch.tensor([[]])

torch.max(input=my_tensor) # Error

my_tensor = torch.tensor([[[]]])

torch.max(input=my_tensor) # Error

import torch

my_tensor = torch.tensor([[]])

torch.max(input=my_tensor, dim=0)
# torch.return_types.max(
# values=tensor([]),
# indices=tensor([], dtype=torch.int64))

my_tensor = torch.tensor([[[]]])

torch.max(input=my_tensor, dim=0)
# torch.return_types.max(
# values=tensor([], size=(1, 0)),
# indices=tensor([], size=(1, 0), dtype=torch.int64))

import torch

tensor1 = torch.tensor([])
tensor2 = torch.tensor([])

torch.max(input=tensor1, other=tensor2)
# tensor([])

tensor1 = torch.tensor([[]])
tensor2 = torch.tensor([[]])

torch.max(input=tensor1, other=tensor2)
# tensor([], size=(1, 0))

tensor1 = torch.tensor([[[]]])
tensor2 = torch.tensor([[[]]])

torch.max(input=tensor1, other=tensor2)
# tensor([], size=(1, 1, 0))

import torch

torch.__version__ # 2.3.0+cu121