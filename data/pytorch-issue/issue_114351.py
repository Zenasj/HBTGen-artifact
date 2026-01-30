import torch.nn as nn

import torch

def func(x, w):
    return torch.nn.functional.conv2d(x, w, groups=int(w.shape[0]))

x = torch.rand(1, 3, 64, 64)
w = torch.rand(3, 1, 3, 3)
y1 = func(x, w)
cfunc = torch.compile(func, fullgraph=True, dynamic=True)
y2 = cfunc(x, w)

torch.testing.assert_close(y1, y2)