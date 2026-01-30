import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        torch.manual_seed(0)
        out = torch.randn(x, y, z, device='cuda')  # CPU is also wrong
        return out


model = Model().eval().cuda()  # CPU is also wrong
c_model = torch.compile(model)

inputs = [1, 1, 1]

for i in range(10):
    print(f"round {i}")
    output = model(*inputs)
    print(output)

c_output = c_model(*inputs)

print(f"after compilation\n{c_output}")
print(torch.allclose(output, c_output))

import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        torch.manual_seed(0)
        out = torch.randn(x, y, z, device='cuda')
        return out


model = Model().eval().cuda()
c_model = torch.compile(model, backend="cudagraphs")

inputs = [1, 1, 1]

for i in range(10):
    print(f"round {i}")
    output = model(*inputs)
    print(output)

c_output = c_model(*inputs)

print(f"after compilation\n{c_output}")
print(torch.allclose(output, c_output))