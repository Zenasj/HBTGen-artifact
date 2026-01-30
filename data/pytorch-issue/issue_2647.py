import os
import torch
import torch.nn as nn

BUG = True
optim_name = 'Adam' # works with SGD

model = nn.Linear(2,1)
optimizer = torch.optim.__dict__[optim_name](model.parameters(), 1e-4)
state_dict = optimizer.state_dict()
path_state = 'state_dict_{}.pth'.format(optim_name)

if not os.path.isfile(path_state):
    torch.save(state_dict, path_state)
else:
    if BUG:
        optimizer.load_state_dict(torch.load(path_state))
    loss = model(torch.autograd.Variable(torch.randn(1,2)))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()