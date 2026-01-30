import torch
import numpy as np

vals = torch.Tensor([nan, 1., nan, 1., nan, nan, nan, inf, 0., nan, 1., nan, nan, nan, 1.], device='cuda:0')
torch.unique(vals, return_counts=True)
np.unique(vals.cpu(), return_counts=True)

(tensor([0., 1., inf, nan, nan, nan, nan, nan, nan, nan, nan, nan], device='cuda:0'), tensor([1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], device='cuda:0'))
(array([ 0.,  1., inf, nan], dtype=float32), array([1, 4, 1, 9]))