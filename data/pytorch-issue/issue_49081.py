import torch
import torch.nn as nn

Result = Pass

model = torch.nn.DataParallel(model, device_ids=(0,) ).cuda()

model = torch.nn.DataParallel(model, device_ids=(1,) ).cuda()