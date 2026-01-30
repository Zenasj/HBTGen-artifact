import torch

In [31]: torch.randperm(5, device='cuda')
Out[31]: tensor([1, 4, 2, 3, 0], device='cuda:0')