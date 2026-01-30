import torch.nn as nn

import torch
p=torch.nn.Parameter(torch.empty(1,device='meta'))
print(p)
#Parameter containing:
#tensor(..., device='meta', size=(1,), requires_grad=True)
pl=torch.nn.ParameterList([p])
print(pl)
#terminate called without an active exception
#[1]    9276 abort (core dumped)  python