import torch.nn as nn

import torch
from torch import nn
from torch.autograd import grad

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.lin1 = nn.Linear(3, 30)
        self.lin2 = nn.Linear(30, 1)

    def forward(self, p):
        x = self.lin1(p)
        x = nn.ReLU()(x)
        return self.lin2(x)

x = torch.randn(100, 3)
y = (5 * torch.sin(x) + 3 * torch.cos(x)).sum(dim=-1).unsqueeze(-1)
z = (5 * torch.cos(x) - 3 * torch.sin(x)).sum(dim=-1).unsqueeze(-1)
model = net()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

for epoch in range(1000):
    model.train()
    x.requires_grad = True
    optimizer.zero_grad()
    output = model(x)
    grad_x = grad(output.sum(), x, retain_graph=True)[0]
    loss_z = nn.MSELoss()(grad_x.sum(dim=-1).unsqueeze(-1), z)
    print(loss_z.grad_fn)  # None
    loss_z.backward()
    optimizer.step()
    print('Loss_z = {:.4f}.'.format(loss_z.item()))