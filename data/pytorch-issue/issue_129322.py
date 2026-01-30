import torch

my_tensor = torch.tensor([3, 5, 1])

torch.repeat_interleave(input=my_tensor, repeats=3)
my_tensor.repeat_interleave(repeats=3)
# tensor([3, 3, 3, 5, 5, 5, 1, 1, 1])

import torch

my_tensor = torch.tensor([3, 5, 1])

torch.repeat_interleave(input=my_tensor) # Error

import torch

my_tensor = torch.tensor([3, 5, 1])

torch.repeat_interleave(my_tensor)
# tensor([0, 0, 0, 1, 1, 1, 1, 1, 2])

import torch

torch.__version__ # 2.3.0+cu121