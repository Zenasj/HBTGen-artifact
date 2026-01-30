import torch.nn as nn

import torch
from learn2learn import clone_module

lr = 0.01
n_updates = 100

# ------------------ Torch Differentiable ------------------
torch.manual_seed(1)
model = torch.nn.Linear(3, 1)
model_clone = clone_module(model)
for param in model_clone.parameters():
    param.retain_grad()
optim = torch.optim.Adam(model_clone.parameters(), lr=lr)

x = torch.rand((n_updates, 3), requires_grad=True)
for i in range(n_updates):
    b_x = x[i]
    y = torch.rand((1,), requires_grad=True)
    out = model_clone(b_x)
    loss = ((out - y) ** 2).sum()
    optim.zero_grad()
    loss.backward(retain_graph=True)
    optim.step()
params_1 = next(model_clone.parameters()).detach()

# ------------------ Torch ------------------
torch.manual_seed(1)
model = torch.nn.Linear(3, 1)
optim = torch.optim.Adam(model.parameters(), lr=lr)

x = torch.rand((n_updates, 3), requires_grad=True)
for i in range(n_updates):
    b_x = x[i]
    y = torch.rand((1,), requires_grad=True)
    out = model(b_x)
    loss = ((out - y) ** 2).sum()
    optim.zero_grad()
    loss.backward()
    optim.step()
params_2 = next(model.parameters()).detach()

print("All close:", torch.allclose(params_1, params_2))
print("Difference:", params_1 - params_2)