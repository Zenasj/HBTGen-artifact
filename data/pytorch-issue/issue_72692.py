import torch.nn as nn

import torch
from torch import nn
import numpy as np
import math

H_in = 10
H_out = 10
num_layers = 2
D = 1

gru = nn.GRU(H_in, H_out, num_layers)
input = torch.randn(1, 1, H_in)
h0 = torch.randn(D * num_layers, 1, H_out)
output, h_n = gru(input, h0)

print(output)
print(h_n)

print(gru._all_weights)

# the same result can be calculated directly
x = input
output = x
h_n = []
for i in range(num_layers):
    h = h0[i]
    W_ih, W_hh, b_ih, b_hh = gru._flat_weights[i * 4 : (i + 1) * 4]
    W_ir, W_iz, W_in = W_ih.split(H_in)
    W_hr, W_hz, W_hn = W_hh.split(H_in)
    b_ir, b_iz, b_in = b_ih.split(H_in)
    b_hr, b_hz, b_hn = b_hh.split(H_in)

    r = torch.sigmoid(x @ W_ir.T + b_ir + h @ W_hr.T + b_hr)
    z = torch.sigmoid(x @ W_iz.T + b_iz + h @ W_hz.T + b_hz)
    n = torch.tanh(x @ W_in.T + b_in + r * (h @ W_hn.T + b_hn))
    h = (1 - z) * n + z * h
    x = h
    output = x
    h_n.append(h[0])

print(output)
print(h_n)