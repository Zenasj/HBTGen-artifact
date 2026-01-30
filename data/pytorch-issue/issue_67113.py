import torch

_0 = torch.sum(torch.diagonal(argument_0, 0, 1, 2), [1])
_1 = torch.diagonal(torch.matmul(argument_0, argument_0), 0, 1, 2)
_2 = torch.sum(_1, [1])
...