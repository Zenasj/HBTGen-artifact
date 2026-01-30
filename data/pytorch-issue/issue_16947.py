import torch.nn as nn

import torch


class NaughtyFun(torch.autograd.Function):
    def forward(self, x):
        self._grad = x.clone()
        return x

    def backward(self, g):
        return self._grad


class MyModule(torch.nn.Module):
    def __init__(self):
        self.fun = NaughtyFun()
        super(MyModule, self).__init__()

    def forward(self, x):
        return self.fun(x)


net = torch.nn.DataParallel(MyModule(), device_ids=[0, 1]).cuda()
output = net(torch.zeros((4, 4), requires_grad=True).cuda())
output.sum().backward()