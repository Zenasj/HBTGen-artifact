import torch

a = b > c
na = a == 0

a = b > c
na = a == 0

a = b > c
na = a == torch.as_tensor(False)