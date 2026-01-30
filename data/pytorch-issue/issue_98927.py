import numpy as np
import torch

for device in [torch.device('cpu'), torch.device('cuda:0')]:
    for dtype in [torch.float32, torch.float64]:
        x = torch.tensor([[[-.3, 1.]]], device = device, dtype = dtype)
        x = torch.nn.Parameter(x)

        y = torch.nn.functional.avg_pool1d(x, 2).reshape(x.size(0), -1)
        y.backward()
        print('x.grad = {}'.format(x.grad))

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Arguments
parser = argparse.ArgumentParser(description = 'bug_avg_pool1d')

parser.add_argument('--cuda', action = 'store_true', default = False, help = 'use cuda')
parser.add_argument('--float64', action = 'store_true', default = False, help = 'use float64 instead of float32')
parser.add_argument('--avg_pool', action = 'store_true', default = False, help = 'use avg_pool instead of max_pool')
parser.add_argument('--with_mse_loss', action = 'store_true', default = False, help = 'use mse_loss instead of no loss')

### Set up parameters
args = parser.parse_args()
use_cuda = torch.cuda.is_available() and args.cuda
device = torch.device("cuda:0" if use_cuda else "cpu")
if args.float64:
    torch.set_default_dtype(torch.float64)
else:
    torch.set_default_dtype(torch.float32)

print('use_cuda: {}'.format(use_cuda))
print('float64: {}'.format(args.float64))
print('avg_pool: {}'.format(args.avg_pool))
print('with_mse_loss: {}'.format(args.with_mse_loss))

# Model definition
img_size = 4
in_channels = 1
mid_channels = 1
out_dim = 1

class Model(torch.nn.Module):
    def __init__(self, avg_pool):
        super(Model, self).__init__()

        self.avg_pool = avg_pool

        self.conv = torch.nn.Conv1d(in_channels, mid_channels, 3, bias = False)
        with torch.no_grad():
            self.conv.weight.fill_(1)

    def forward(self, x):
        x = self.conv(x)

        if self.avg_pool:
            x = torch.nn.functional.avg_pool1d(x, 2)
        else:
            x = torch.nn.functional.max_pool1d(x, 2)

        x = x.reshape(x.size(0), -1)

        return x

# Build dataset
dataset_size = 1

x = torch.ones(dataset_size, in_channels, img_size, device = device)
y = torch.zeros(dataset_size, out_dim, device = device)

# Instantiate model
model = Model(args.avg_pool)
if use_cuda: model = model.cuda()

f_loss = nn.MSELoss(reduction = 'mean') if args.with_mse_loss else lambda y1, y2: y1

# Differentiate with grad
loss = f_loss(model(x), y)
print('loss = {}'.format(loss))
jac_grad = torch.autograd.grad(loss, model.conv.weight)
print('autograd grad of conv.weight = {}'.format(jac_grad))

# Differentiate with backward
model.zero_grad()
loss = f_loss(model(x), y)
loss.backward()
print('backward grad of conv.weight = {}'.format(model.conv.weight.grad))