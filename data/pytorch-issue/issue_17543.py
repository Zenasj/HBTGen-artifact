import torch.nn as nn

import torch
rnn = torch.nn.LSTM(10,10)  # same error with e.g. torch.nn.GRU(10,10,1)
rnn.cuda()

import torch
rnn = torch.nn.LSTM(10,10)
try:
    rnn.cuda()
except:
    rnn.cuda()