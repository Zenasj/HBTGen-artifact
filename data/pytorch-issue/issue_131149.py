import torch

my_tensor = torch.tensor([]) # 1D tensor

torch.nanmedian(input=my_tensor)
# tensor(nan)

my_tensor = torch.tensor([[]]) # 2D tensor

torch.nanmedian(input=my_tensor)
# tensor(nan)

my_tensor = torch.tensor([[[]]]) # 3D tensor

torch.nanmedian(input=my_tensor)
# tensor(nan)

import torch

my_tensor = torch.tensor([]) # 1D tensor

torch.nanmedian(input=my_tensor, dim=0) # Error

my_tensor = torch.tensor([[]]) # 2D tensor

torch.nanmedian(input=my_tensor, dim=1) # Error

my_tensor = torch.tensor([[[]]]) # 3D tensor

torch.nanmedian(input=my_tensor, dim=2) # Error

import torch

torch.__version__ # 2.3.1+cu121