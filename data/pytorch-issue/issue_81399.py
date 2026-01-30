import torch

torch.randn(2).unflatten(0, (('B1', -1), ('B2', 1)))

torch.randn(2, names=('A',)).unflatten('A', (('B1', -1), ('B2', 1)))