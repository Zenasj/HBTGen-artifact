import torch

my_tensor = torch.tensor([8, -3, 0, 1, 5, -2, -1, 4])
                                         # ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓
torch.index_select(input=my_tensor, dim=0, index=torch.tensor([[0, 2, 4]])) # Error

import torch

my_tensor = torch.tensor([8, -3, 0, 1, 5, -2, -1, 4])
                                         # ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓
torch.index_select(input=my_tensor, dim=0, index=torch.tensor(4))
# tensor([5])

import torch

torch.__version__ # '2.3.0'