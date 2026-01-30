import torch
import torch.nn as nn

m = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Sigmoid()).cuda()
x = torch.randn(4, 4, device='cuda')
m = torch.compile(m) 
y = m(x)         
y.sum().backward()