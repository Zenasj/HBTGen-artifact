import torch.nn as nn

# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.cuda.amp import autocast
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

rnn = nn.GRU(input_size=768, hidden_size=512, batch_first=True,
                                 bidirectional=True, num_layers=3, dropout=0.5).to(device)

inputs = torch.randn(10, 231, 768).to(device)
h0 = torch.randn(2, 3, 768).to(device)
print('Allocated after init:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')

with autocast():
    output, hn = rnn(inputs)
    print(output.dtype)
        
print('Allocated after rnn pass:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
print('Size of output:', sys.getsizeof(output.storage()))
print('Size of hn:', sys.getsizeof(hn.storage()))