import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(torch.jit.ScriptModule):

    def __init__(self):
        super(Net, self).__init__()

        self.conv = nn.Conv2d(1, 1, 1)

        self.lin = nn.Linear(100, 1)

    @torch.jit.script_method
    def forward(self, x):
        x = F.relu(self.conv(x))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.lin(x))

        return x

m = Net()

x = torch.ones((1, 1, 10, 10))

for i in range(1000000):
    m(x)

import torch
import torch.nn.functional as F
import time

weight = torch.ones([1,1,1,1])
x = torch.ones((1, 1, 10, 10))

oo = (1, 1)
zz = (0, 0)

start = time.time()
with torch.no_grad():
    while time.time() - start < 10:
        F.conv2d(x, weight, None, oo, zz, oo, 1)