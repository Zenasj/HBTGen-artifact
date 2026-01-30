import torch.nn as nn

import torch
from torch import nn
rnn = nn.LSTMCell(10, 20)
input = torch.randn(3, 10)
hx = torch.randn(3, 20)
cx = torch.randn(3, 20)
output = []
for i in range(6):
  hx, cx = rnn(input[i], (hx, cx))
  output.append(hx)