out = b1(x) + (alpha * b2(x))

import torch
import numpy as np
import torch.nn as nn

alpha = 1
channels = 64

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # conv = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.b1 = nn.Sequential(torch.nn.ReLU())

        # conv = torch.nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=False)
        self.b2 = nn.Sequential(torch.nn.ReLU())

        self.l = nn.Linear(channels, 10)

    def forward(self, x):
        x.register_hook(lambda g: print(f"[IN] V = {g.var():.5e}, M = {g.mean():.5e}"))

        b1 = self.b1(x)
        b1.register_hook(lambda g: print(f"[B1] V = {g.var():.5e}, M = {g.mean():.5e}"))

        b2 = self.b2(x)
        b2.register_hook(lambda g: print(f"[B2] V = {g.var():.5e}, M = {g.mean():.5e}"))

        out = torch.add(b1, b2, alpha=alpha) # out = b1 + (alpha * b2)
        out.register_hook(lambda g: print(f"[OUT] V = {g.var():.5e}, M = {g.mean():.5e}"))

        out = nn.AdaptiveMaxPool2d(1)(out)
        out = nn.Flatten()(out)
        return self.l(out)


y = torch.randint(low=0, high=10, size=[128])
x = torch.normal(mean=0, std=1, size=[128, channels, 32, 32], requires_grad=True)

net = Net()
out = net(x)
loss = torch.nn.CrossEntropyLoss()(out, y)
loss.backward()