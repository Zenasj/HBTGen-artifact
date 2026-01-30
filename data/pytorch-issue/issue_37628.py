import torch

torch.einsum('ij,j', torch.randn(0, 3), torch.randn(3))