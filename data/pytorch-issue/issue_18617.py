import torch.nn as nn

import torch
from torch import nn

device = torch.device('cuda')

rnn = nn.GRU(input_size=256,
    hidden_size=128,
    num_layers=2,
    batch_first=False,
    dropout=0.1,
    bidirectional=True)

rnn.eval()
rnn = rnn.to(device)

with torch.no_grad():
    traced_rnn = torch.jit.trace(rnn, torch.rand((50, 10, 256), dtype=torch.float32).to(device))

params = {}

for k, v in traced_rnn.named_parameters():
    params[k] = v
    
torch.jit.save(traced_rnn, 'traced.pt')
traced_rnn = torch.jit.load('traced.pt')

for k, v in traced_rnn.named_parameters():
    diff = torch.max(torch.abs(params[k] - v)).item()
    
    if diff > 0:
        print(k, diff)