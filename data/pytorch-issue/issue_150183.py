import torch
import torch.nn as nn
model = nn.Sequential(nn.Linear(5, 10), nn.ReLU(), nn.Linear(10, 5), nn.ReLU())
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, differentiable=True)
# optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, differentiable=True)
x = torch.randn(100, 5)
y = torch.randn(100, 5)
criterion = nn.MSELoss()
optimizer.zero_grad()
loss = criterion(model(x), y)
loss.backward()
optimizer.step()

import torch
import torch.nn as nn
model = nn.Sequential(nn.Linear(5, 10), nn.ReLU(), nn.Linear(10, 5), nn.ReLU())
params = torch.cat([p.view(-1) for p in model.parameters()])*1
params.retain_grad()
lr = torch.tensor(0.01, requires_grad=True)
optimizer = torch.optim.Adam([params], lr=lr, differentiable=True)
# optimizer = torch.optim.AdamW([params], lr=lr, differentiable=True)
x = torch.randn(100, 5)
y = torch.randn(100, 5)
criterion = nn.MSELoss()
optimizer.zero_grad()
loss = criterion(model(x), y)
loss.backward()
optimizer.step()