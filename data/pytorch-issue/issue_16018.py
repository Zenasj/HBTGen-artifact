import torch.nn as nn
import random

import numpy as np
import torch
from torch import nn

d_in = 25
d_out = 64
k = 7
B = 2
L = 10

conv = nn.Conv1d(d_in, d_out, k, stride=2, padding=0)
pad = nn.ConstantPad1d((2, 3), 0)
np.random.seed(1034)
X = np.random.random((B, L, d_in))
X = torch.as_tensor(X).transpose(-1, -2)
X = pad(X)
w = np.random.random((k, d_in, d_out))
w = w.astype('float32')
b = np.random.random((d_out, ))

w = torch.as_tensor(w).permute(2, 1, 0).float()
b = torch.as_tensor(b).float()

conv.weight = nn.Parameter(w)
conv.bias = nn.Parameter(b)

out1 = conv(X.float()).detach().numpy()
w1 = conv.weight.detach().numpy()
out2 = conv.double()(X).detach().numpy()
w2 = conv.weight.detach().numpy()
out3 = conv.float()(X.float()).detach().numpy()
w3 = conv.weight.detach().numpy()
print('w == w1?')
print(np.allclose(w.detach().numpy(), w1))
print('w1 == w2?')
print(np.allclose(w1, w2))
print('w2 == w3?')
print(np.allclose(w2, w3))
print('out1 == out2?')
print(np.allclose(out1, out2, atol=1e-6))
print('out2 == out3?')
print(np.allclose(out2, out3, atol=1e-6))

import numpy as np
import torch
from torch import nn

d_in = 25
d_out = 64
k = 7
B = 2
L = 10

conv = nn.Conv1d(d_in, d_out, k, stride=2, padding=0)
pad = nn.ConstantPad1d((2, 3), 0)
np.random.seed(1034)
X = np.random.random((B, L, d_in))
X = torch.as_tensor(X).transpose(-1, -2).cuda()
X = pad(X)
w = np.random.random((k, d_in, d_out))
w = w.astype('float32')
b = np.random.random((d_out, ))

w = torch.as_tensor(w).permute(2, 1, 0).float().cuda()
b = torch.as_tensor(b).float().cuda()

conv.weight = nn.Parameter(w)
conv.bias = nn.Parameter(b)

out1 = conv(X.float()).detach().cpu().numpy()
w1 = conv.weight.detach().cpu().numpy()
out2 = conv.double()(X).detach().cpu().numpy()
w2 = conv.weight.detach().cpu().numpy()
out3 = conv.float()(X.float()).detach().cpu().numpy()
w3 = conv.weight.detach().cpu().numpy()
print('w == w1?')
print(np.allclose(w.detach().cpu().numpy(), w1))
print('w1 == w2?')
print(np.allclose(w1, w2))
print('w2 == w3?')
print(np.allclose(w2, w3))
print('out1 == out2?')
print(np.allclose(out1, out2, atol=1e-6))
print('out2 == out3?')
print(np.allclose(out2, out3, atol=1e-6))