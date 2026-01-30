import torch

input_size = 5
hidden_size = 6
rnn = torch.nn.GRU(input_size, hidden_size)

for seq_len in reversed(range(4)):
    output, h_n = rnn(torch.zeros(seq_len, 10, input_size))
    print('{}, {}'.format(output.shape, h_n.shape))

import torch

input_size = 5
hidden_size = 6
rnn = torch.nn.LSTM(input_size, hidden_size, bidirectional=True)
output, h_n = rnn(torch.zeros(0, 10, input_size))

import torch
import torch.nn.utils.rnn as rnn_utils
import torch.nn as nn
packed = rnn_utils.pack_sequence([])