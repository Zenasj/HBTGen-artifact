import torch

a = torch.ones(1000, 1000)
b = a[0].clone()  # we're cloning the tiny 1000-element tensor, instead of the 1mil element input