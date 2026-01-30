import torch.nn as nn

py
import torch

model = torch.nn.Linear(10, 10).cuda()
model = torch.nn.DataParallel(model, device_ids=[0, 1])
output = model(torch.randn(20, 10).cuda())

import torch

model = torch.nn.Linear(10, 10).cuda()
model = torch.nn.DataParallel(model, device_ids=[0, 0])
output = model(torch.randn(20, 10).cuda())