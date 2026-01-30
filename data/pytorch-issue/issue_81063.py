import torch.nn as nn

import torch
import io


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

    def forward(self, x, y):
        x = torch.nn.functional.relu(x)
        z1 = x + y
        z2 = torch.sum(z1, 0)
        return z1, z2


example_args = (torch.randn(8, 8), torch.randn(8))

net = Net()
traced = torch.jit.trace(net, example_args)
print('traced graph\n', traced.graph)
print('traced graph after serialization\n', torch.jit.load(io.BytesIO(traced.save_to_buffer())).graph)