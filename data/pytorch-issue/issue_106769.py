import torch.nn as nn

import torch
m = torch.nn.GRUCell(1,1)
input = torch.randn(1,1)
hx = torch.randn(1,1,5,127,1)
hx = m(input,hx)

import torch
m = torch.nn.LSTMCell(1,1)
input = torch.randn(1,1)
hx = torch.randn(1,1,5,127,1)
cx = torch.randn(1,1)
hx, cx = m(input,(hx,cx))