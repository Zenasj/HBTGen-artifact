import torch.nn as nn

py
import torch

torch.manual_seed(420)

class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        n = x.shape[0]
        v = x.split(2)
        y = torch.zeros([4, 2, 2, 3])
        z = [i + 1 for i in range(n)]
        y[z] = v[0]
        return y


func = Model().to('cuda')

x = torch.randn(2, 2, 3).to('cuda')

with torch.no_grad():
    func = func.eval()

    jit_func = torch.compile(func)
    res2 = jit_func(x)
    # segmentation fault (core dumped)