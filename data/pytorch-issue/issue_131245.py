import torch

my_tensor = torch.tensor([]) # 1D tensor

torch.mode(input=my_tensor, dim=0) # Error

my_tensor = torch.tensor([[]]) # 2D tensor

torch.mode(input=my_tensor, dim=1) # Error

my_tensor = torch.tensor([[[]]]) # 3D tensor

torch.mode(input=my_tensor, dim=2) # Error

import torch

my_tensor = torch.tensor([[]]) # 2D tensor

torch.mode(input=my_tensor, dim=0)
# torch.return_types.mode(
# values=tensor([]),
# indices=tensor([], dtype=torch.int64))

my_tensor = torch.tensor([[[]]]) # 3D tensor

torch.mode(input=my_tensor, dim=0)
torch.mode(input=my_tensor, dim=1)
# torch.return_types.mode(
# values=tensor([], size=(1, 0)),
# indices=tensor([], size=(1, 0), dtype=torch.int64))

import torch

torch.__version__ # 2.3.1+cu121