import torch.nn as nn

import torch


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, y):
        return torch.asinh(y)


x = torch.tensor([0, 0, 0, -10000.1])  # triggering condition: `shape >= 4` and `x[3] < -10000.0`

m = Model()
c_m = torch.compile(m)

output = m(x)
c_output = c_m(x)

print(output)
print(c_output)