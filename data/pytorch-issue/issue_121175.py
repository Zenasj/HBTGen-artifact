import torch.nn as nn

py
import torch

torch.manual_seed(42)

class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.weight1 = torch.nn.Parameter(torch.randn(10, 10, dtype=torch.float64))
        self.weight2 = torch.nn.Parameter(torch.randn(10, 10, dtype=torch.float64))
        self.bias = torch.nn.Parameter(torch.randn(10, dtype=torch.float64))

    def forward(self, x1):
        v1 = torch.mm(x1, self.weight1)
        v2 = torch.addmm(self.bias, x1, self.weight2)
        return (v1, v2)


func = Model().to('cpu')

x = torch.randn(10, 10, dtype=torch.float64)

with torch.no_grad():
    print(func(x.clone()))

    func1 = torch.compile(func)
    print(func1(x.clone()))