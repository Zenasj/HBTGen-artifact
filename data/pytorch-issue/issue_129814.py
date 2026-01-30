import torch

my_tensor = torch.tensor([0, 1, 2], device='cpu')

# With `print()`
print(my_tensor.device) # cpu

# Without `print()`
my_tensor.device # device(type='cpu')

my_tensor = torch.tensor([0, 1, 2], device='cuda:0')

# With `print()`
print(my_tensor.device) # cuda:0

# Without `print()`
my_tensor.device # device(type='cuda', index=0)

import torch

my_tensor = torch.tensor([0, 1, 2], dtype=torch.int32)

# With `print()`
print(my_tensor.dtype) # torch.int32

# Without `print()`
my_tensor.dtype # torch.int32

import torch

torch.__version__ # 2.3.0