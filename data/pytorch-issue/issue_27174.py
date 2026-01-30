import torch

x = torch.randn(2, 3, 5, 7)
x.refine_names(..., 'D')
x.align_to('D', ...)