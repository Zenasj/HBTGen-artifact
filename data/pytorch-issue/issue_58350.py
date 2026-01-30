import torch.nn as nn

import torch

print(list(torch.nn.LazyLinear(10, 5).named_parameters()))

"""
[('weight', Uninitialized parameter), ('bias', Parameter containing:
tensor([ 0.0000e+00, -2.5244e-29, -9.3202e+27, -2.0005e+00,  1.1210e-44,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00],
       requires_grad=True))]
"""

print(list(torch.nn.LazyConv2d(10, 5, 5).named_parameters()))

"""
[('weight', Uninitialized parameter), ('bias', Parameter containing:
tensor([ 0.0000e+00, -2.5244e-29, -9.3202e+27, -2.0005e+00,  1.1210e-44,
        -0.0000e+00,  0.0000e+00,  0.0000e+00,  3.5873e-43,  3.6013e-43],
       requires_grad=True))]
"""