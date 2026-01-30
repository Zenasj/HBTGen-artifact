import torch.nn as nn

import torch
from torch import nn

T = 10
F = 20
device = torch.device('cuda')

print('Generating data')
data = (torch.rand(1, T, F) * 0.1).to(device)

print('Loading model')
model = nn.LSTM(F, F, num_layers=1, batch_first=True, bidirectional=True, dropout=0)
model = model.eval().to(device)

print('Tracing model')
tmodel = torch.jit.trace(model, (data,))
tmodel.save('/tmp/test.pt')

print('Productionazing model')
pmodel = torch.jit.load('/tmp/test.pt', map_location=device)

print('Forwarding data')
with torch.no_grad():
    o1 = model(data)[0]
    o2 = tmodel(data)[0]
    o3 = pmodel(data)[0]

assert (o1 == o2).all()   # WORKS
assert (o2 == o3).all()   # FAILS