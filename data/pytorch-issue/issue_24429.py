import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(1, 1, 3)

    def forward(self, x):
        return self.conv(x)

    def weighted_kernel_sum(self, weight):
        return weight * self.conv.weight

example_weight = torch.rand(1, 1, 3, 3)
example_forward_input = torch.rand(1, 1, 3, 3)
n = Net()
# the following two calls are equivalent
module = torch.jit.trace_module(n, example_forward_input)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(1, 1, 3)

    def forward(self, x):
        return self.conv(x)

    def weighted_kernel_sum(self, weight):
        return weight * self.conv.weight

example_forward_input = torch.rand(1, 1, 3, 3)
n = Net()
inputs = {'forward' : example_forward_input}
module = torch.jit.trace_module(n, inputs)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(1, 1, 3)

    def forward(self, x):
        return self.conv(x)

    def weighted_kernel_sum(self, weight):
        return weight * self.conv.weight

example_weight = torch.rand(1, 1, 3, 3)
example_forward_input = torch.rand(1, 1, 3, 3)
n = Net()
inputs = {'forward' : example_forward_input, 'weighted_kernel_sum' : example_weight}
module = torch.jit.trace_module(n, inputs)