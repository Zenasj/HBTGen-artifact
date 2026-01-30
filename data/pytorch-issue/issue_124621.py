import torch
import torch.nn as nn

lf = nn.MSELoss(reduction='none')
device = 'mps'

model = nn.Sequential(
    nn.Conv1d(3, 3, 1),
)

model = model.to(device)

x = torch.randn(128, 10, 3).to(device)
x = x.permute(0, 2, 1)
y = model(x)
y = y.permute(0, 2, 1)[:,:5,:]
y_hat = torch.randn(128, 5, 3).to(device)

loss = lf(y, y_hat)

if (loss < 0).any():
    print('negative')

# output: negative

import torch
import torch.nn as nn

lf = nn.MSELoss(reduction='none')
device = 'cpu'

model = nn.Sequential(
    nn.Conv1d(3, 3, 1),
)

model = model.to(device)

x = torch.randn(128, 10, 3).to(device)
x = x.permute(0, 2, 1)
y = model(x)
y = y.permute(0, 2, 1)[:,:5,:]
y_hat = torch.randn(128, 5, 3).to(device)

loss = lf(y, y_hat)

if (loss < 0).any():
    print('negative')

# no negative report