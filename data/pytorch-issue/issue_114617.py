import torch
import torch.nn as nn

rnn = nn.RNN(2, 3, num_layers=1, nonlinearity='relu')
x = torch.randn(5, 1, 2)
rnn(x)