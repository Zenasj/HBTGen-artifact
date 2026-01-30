import torch

my_tensor = torch.tensor([7, 4, 5])

my_tensor.repeat(sizes=(3, 2)) # Error

import torch

my_tensor = torch.tensor([7, 4, 5])

my_tensor.repeat(repeats=(3, 2))
# tensor([[7, 4, 5, 7, 4, 5],
#         [7, 4, 5, 7, 4, 5],
#         [7, 4, 5, 7, 4, 5]])