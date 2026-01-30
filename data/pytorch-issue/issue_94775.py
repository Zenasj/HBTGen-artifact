import torch.nn as nn

py
import torch

torch.manual_seed(420)
x = torch.randn(1, 1, 6, 7)

def func(x):
    y = torch.nn.functional.max_pool2d(x, 1, stride=(2, 2), padding=0, ceil_mode=True) 
    return y

print(func(x))
# tensor([[[[-0.0070,  0.6704,  0.0302, -0.5131],
#           [ 2.0182, -0.2523, -0.0646,  0.0938],
#           [ 0.8388, -0.5796,  1.4694, -1.8402]]]])

print(torch.compile(func)(x))
# tensor([[[[-0.0070,  0.6704,  0.0302, -0.5131],
#           [ 2.0182, -0.2523, -0.0646,  0.0938],
#           [ 0.8388, -0.5796,  1.4694, -1.8402],
#           [   -inf,    -inf,    -inf,    -inf]]]])