import torch.nn as nn

import torch
import torch._inductor.config as config
config.trace.enabled = True
torch._dynamo.config.verbose=True

class HelloModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1).cuda()

    def forward(self, x):
        x = self.linear(x)
        x_2 = torch.relu(x)
        y = torch.relu(x_2)
        return x_2 + y

m = HelloModule()

@torch.compile
def f():
    opt = torch.optim.Adam(m.parameters(), lr=0.01)
    for i in range(5):
        opt.zero_grad()
        out = m(torch.ones(1).to(device='cuda:0'))
        loss = out.sum()
        loss.backward()
        opt.step()

f()