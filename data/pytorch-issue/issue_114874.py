import torch.nn as nn

import torch
from torch import nn
seq_length = 5
batch_size = 3
rnn=nn.LSTM(input_size=13,hidden_size=4,num_layers = 2,batch_first=True)
input_sequence = torch.rand((batch_size, seq_length, 2))
rnn(input_sequence)