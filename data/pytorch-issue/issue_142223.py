import torch
import torch.nn as nn

torch.manual_seed(42)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.tanh(x)
        return x


m = Model()
x = torch.randn(64, 64)

with torch.no_grad():
    m = m.eval()
    y = m(x)

    c_m = torch.compile(m)
    c_y = c_m(x)

    print(torch.allclose(y, c_y))
    print(torch.max(torch.abs(y - c_y)))