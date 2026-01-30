import torch
import torch.nn as nn

model1 = torch.nn.DataParallel(Net(), device_ids=[0,1]).cuda()
model2 = torch.nn.DataParallel(Net(), device_ids=[2,3]).cuda()

def task(model, data):
  ...
  return out

out1 = task(model1, data1)
out2 = task(model2, data2)