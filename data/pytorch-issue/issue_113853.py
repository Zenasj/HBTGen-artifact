import torch.nn as nn

conv_forward = _conv.forward
x = torch.randn((3, 4, 5, 5)).to("cuda")
w = torch.rand((2, 4, 3, 3)).to("cuda")
b = torch.rand(2).to("cuda")
x = conv_forward(x, w, b, padding=(1, 1))
print(x)

import torch
import torch._inductor 

@torch.compile(mode='max-autotune')
def conv_forward(x, w, b):
    return torch.nn.functional.conv2d(x, w, b, padding=(1, 1))


x = torch.randn((3, 4, 5, 5)).to("cuda")
w = torch.rand((2, 4, 3, 3)).to("cuda")
b = torch.rand(2).to("cuda")
x = conv_forward(x, w, b)