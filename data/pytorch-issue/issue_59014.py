import torch

torch.add(a, a, out=a.view([1, *a.shape]))