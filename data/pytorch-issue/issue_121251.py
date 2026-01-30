import torch.nn as nn

py
import torch

torch.manual_seed(42)

class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x1):
        v = torch.index_select(x1, 0, torch.tensor([2]))
        return v


func = Model().to('cpu')

x = torch.randn(2, 2, 2, 2)

with torch.no_grad():
    func1 = torch.compile(func)
    print(func1(x.clone()))
    # tensor([[[[1.0743e+23, 3.0618e-41],
    #       [9.1084e-44, 0.0000e+00]],

    #      [[3.2230e-44, 0.0000e+00],
    #       [3.2230e-44, 0.0000e+00]]]])

    print(func(x.clone()))
    # IndexError: index out of range in self