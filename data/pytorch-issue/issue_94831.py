py
import torch

x = torch.randn(2, 3)

def func(x):
    x.add_(1)
    x.resize_as_(torch.randn(3, 4))
    return x

torch.compile(func)(x)
# IOT instruction (core dumped) or segmentation fault (core dumped)