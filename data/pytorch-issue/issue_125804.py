import torch.nn as nn

import torch
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28, 1)

    def forward(self, x):
        output = self.fc1(x)
        return output


x = torch.rand(28, 28, device="cuda")
model = Net().to(device="cuda")
x_pt2 = torch.compile(model, mode="max-autotune")(x)
try:
    torch._assert_async(torch.tensor(0, device="cuda"))
except:
    print("ignoring exception")
    
# check for `Aborted (core dumped)` on process exit

import torch
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28, 1)

    def forward(self, x):
        output = self.fc1(x)
        return output


x = torch.rand(28, 28, device="cuda")
model = Net().to(device="cuda")
x_pt2 = torch.compile(model, mode="max-autotune")(x)
try:
    torch._assert_async(torch.tensor(0, device="cuda"))
except:
    print("ignoring exception")
    
# check for `Aborted (core dumped)` on process exit