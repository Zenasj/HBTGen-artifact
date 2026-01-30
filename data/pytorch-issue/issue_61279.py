import torch.nn as nn

import torch
from torch import vmap
import torch.nn.functional as F

class vmapTensor(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func == F.linear:
            func = vmap(func, (0, 0), 0)
        return super().__torch_function__(func, types, args, kwargs)

x = vmapTensor(torch.randn(10, 5, 10))
W = vmapTensor(torch.randn(10, 5, 10))

print(F.linear(x, W))

x = MagicTensor(torch.ones(3,4)).refine_names('A', 'B')
y = MagicTensor(torch.ones(4,5)).refine_names('B', 'C')