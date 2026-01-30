import torch
import torch.nn as nn

model = nn.Sequential(nn.Linear(1024, 1024, bias=False), nn.Linear(1024, 1024, bias=False)).cuda()
x = torch.rand(4, 256, 1024).cuda()
y = model(x).sum()
memory1 = torch.cuda.memory_allocated()
y.backward()
memory2 = torch.cuda.memory_allocated()