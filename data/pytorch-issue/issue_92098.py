import torch.nn as nn

py
import torch
t = torch.zeros(1,3,3).to('mps')
nn.init.eye_(t[0]) # this leaves the t set to zeros
print('eye on mps:\n', t[0])  

t = torch.zeros(1,3,3).to('cpu')
nn.init.eye_(t[0]) # this works
print('eye on cpu:\n', t[0])