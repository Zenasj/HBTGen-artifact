import torch.nn as nn

import torch
model = torch.nn.Sequential(torch.nn.Linear(1, 1))
optim = torch.optim.Adagrad(model.parameters())
loss = torch.nn.MSELoss()
inp = torch.tensor([[45.0]], device="cuda")
outp = torch.tensor([[99.0]], device="cuda")

model.to(device="cuda")
optim.zero_grad()
loss(model(inp), outp).backward()
optim.step()