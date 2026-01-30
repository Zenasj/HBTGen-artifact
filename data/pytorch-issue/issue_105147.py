import torch.nn as nn

import torch
import gc


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


def foo():
    net = Net().cuda()
    x = torch.tensor([[2.0]]).cuda()
    y = net(x)

    a = torch.ones((1, 1)).cuda()

    a += a * y # change to a += y no memory leak


gc.collect()
print(torch.cuda.memory_allocated(0)) # output: 0
foo()
gc.collect()
print(torch.cuda.memory_allocated(0)) # output: 2048