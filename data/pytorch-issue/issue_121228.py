import torch.nn as nn

py
import torch

class Model(torch.nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x1):
        v1 = torch.unbind(x1, dim=self.dim)
        v2 = v1[1]
        v3 = torch.stack([v1[0], v2], dim=self.dim)
        return v3

func = Model(1).to('cpu')

x = torch.arange(12).view(3, 4).float()

with torch.no_grad():
    print(func(x.clone()))
    # tensor([[0., 1.],
    #     [4., 5.],
    #     [8., 9.]])

    func1 = torch.compile(func)
    print(func1(x.clone()))
    # tensor([[ 0.,  1.,  2.,  3.],
    #     [ 4.,  5.,  6.,  7.],
    #     [ 8.,  9., 10., 11.]])