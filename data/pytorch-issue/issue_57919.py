import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = nn.Linear(1, 1, bias=False)

    def forward(self, x):
        return self.l(x)


x = torch.ones((1, 1), dtype=torch.float32, device='cuda:0')
n = Net()
n.to('cuda:0')
print(n.state_dict())
with torch.no_grad():
    print(f'x = {x}, n(x) = {n(x)}')
n.to('cpu')
x = x.to('cpu')
print(n.state_dict())
with torch.no_grad():
    print(f'x = {x}, n(x) = {n(x)}')

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.l0 = torch.nn.Linear(1, 30)
        self.l1 = torch.nn.Linear(30, 30)
        self.l2 = torch.nn.Linear(30, 1)

    def forward(self, x):
        print(x)
        x = F.relu(self.l0(x))
        print(x.view(-1).detach().cpu().numpy())
        x = F.relu(self.l1(x))
        print(x.view(-1).detach().cpu().numpy())
        x = self.l2(x)
        return x


if len(sys.argv) == 2 and sys.argv[1] == '--cpu':
    device = torch.device('cpu')
else:
    device = torch.device('cuda:0')

x = torch.ones([1, 1], dtype=torch.float32, device=device)

net = Net()
net.to(device)
net.load_state_dict(torch.load('state.pth', map_location=device))
with torch.no_grad():
    print(net(x).detach().cpu().numpy())

print({x: (y.max(), y.min()) for x, y in torch.load('state.pth', map_location='cpu').items()})