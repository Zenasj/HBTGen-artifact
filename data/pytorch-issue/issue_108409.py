import torch
import torch.nn as nn

x = torch.randn(2, 3, requires_grad=True).to('mps:0')
y = torch.randn(2, 3, requires_grad=True).to('mps:0')
t = torch.randn(2, 3, requires_grad=True).to('mps:0')

fc1 = torch.nn.Linear(3, 3).to('mps:0')
fc2 = torch.nn.Linear(3, 3).to('mps:0')
fc3 = torch.nn.Linear(3, 3).to('mps:0')
model = nn.Sequential(fc1, fc2, fc3)

o = model(x)
o = ((t-o)**2).sum()
g = torch.autograd.grad(o, model.parameters(), create_graph=True)

o2 = model(y)
o2 = ((t-o2)**2).sum()
g2 = torch.autograd.grad(o2, model.parameters(), create_graph=True)

loss = 0
for i in range(len(g)):
    loss += ((g[i] - g2[i]) ** 2).sum()
loss.backward()