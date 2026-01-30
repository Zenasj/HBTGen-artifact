import torch.nn as nn

py
import torch

torch.manual_seed(42)

class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x1):
        v1 = torch.split(x1, [3, 3, 3], dim=-1)
        v2 = torch.stack(v1, dim=-1)
        v3 = torch.tanh(v2)
        return v3

func = Model().to('cpu')

x = torch.randn(1, 9)

with torch.no_grad():
    print(func(x.clone()))
    # tensor([[[ 0.3245,  0.2263,  0.9761],
    #      [ 0.1281, -0.8086, -0.5635],
    #      [ 0.2303, -0.1842,  0.4314]]])

    func1 = torch.compile(func)
    print(func1(x.clone()))
    # tensor([[[ 0.3245,  0.1281,  0.2303],
    #      [ 0.2263, -0.8086, -0.1842],
    #      [ 0.9761, -0.5635,  0.4314]]])