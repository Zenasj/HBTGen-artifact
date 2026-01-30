import torch.nn as nn

import torch

torch.set_default_dtype(torch.float64)
device = "cuda"
model = torch.nn.Linear(200, 1, bias=True).to(device)
opt = torch.optim.Adam(model.parameters(),lr=0.001) 
x = torch.rand(1, 200).to(device)
model(x).sum().backward()
opt.step()