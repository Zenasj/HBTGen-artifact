import torch

(x @ torch.ones(2)).mean().backward()

(x @ torch.ones(2, 1)).sum().backward()