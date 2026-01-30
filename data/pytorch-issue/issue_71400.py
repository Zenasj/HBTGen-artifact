import torch.nn as nn

import torch

input = torch.randn(100, 100, 100).to('cuda:1')
rnn = torch.nn.LSTM(100, 100).to('cuda:1')
out = rnn(input)