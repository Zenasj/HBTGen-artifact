# torch.rand(1, 1, 22, 54, 1, dtype=torch.float32, device='cuda')  # Inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.p0 = torch.tensor(-0.2727, device='cuda')
        self.p1 = torch.randn(1, 1, 22, 54, 41, dtype=torch.float32, device='cuda')

    def forward(self, x):
        atan = torch.atan(self.p0)
        div = torch.div(self.p1, atan)
        tan = torch.tan(x)
        mul = torch.mul(div, tan)
        sub = torch.sub(mul, mul)
        argmax = sub.argmax(3)
        return (mul, argmax)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1, 22, 54, 1, dtype=torch.float32, device='cuda')

