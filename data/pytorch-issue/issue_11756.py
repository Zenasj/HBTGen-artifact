import torch
import torch.nn as nn

testdata = torch.rand(12,3,112,112)
model = torch.nn.DataParallel(model.cuda(), device_ids=[0,1,2,3])
out = model(testdata)

testdata = torch.rand(12,3,112,112)
model = torch.nn.DataParallel(model, device_ids=[0,1,2,3])
out = model(testdata)