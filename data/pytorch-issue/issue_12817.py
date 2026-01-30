import torch.nn as nn

import torch

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv1 = torch.nn.Conv1d(5, 10, 4)
        self.conv2 = torch.nn.Conv1d(10, 5, 3)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)

        return y

m = Model().cuda()
good = torch.randn((1, 5, 10)).cuda()
small = torch.randn((1, 5, 5)).cuda()

traced = torch.jit.trace(m, good)

traced(small)