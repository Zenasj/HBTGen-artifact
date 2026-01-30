import torch
import torch.nn as nn

model = nn.Linear(2, 2)
net = nn.DataParallel(model, device_ids=[0,1])
input_var = torch.randn(10, 2)
net(input_var)