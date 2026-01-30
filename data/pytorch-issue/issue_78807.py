import torch.nn as nn

import torch
import copy

x = torch.rand(1, 1)
L = torch.nn.Linear(1, 1)
adam = torch.optim.Adam(L.parameters(), foreach=False, amsgrad=True)

L2 = copy.deepcopy(L)
adam2 = torch.optim.Adam(L2.parameters(), foreach=True, amsgrad=True)
for i in range(2):
    y = L(x).sum()
    adam.zero_grad()
    y.backward()
    adam.step()
print(adam.state)

for i in range(2):
    y = L2(x).sum()
    adam2.zero_grad()
    y.backward()
    adam2.step()
print(adam2.state)