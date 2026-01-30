import torch
import torch.nn as nn

def splitA(x):
    return torch.split(x, 3)
class NetA(torch.nn.Module):
    def forward(self, x):
        return splitA(x)
print(torch.jit.script(NetA())) # pass

########

def splitB(x, split_size):
    return torch.split(x, split_size)
class NetB(torch.nn.Module):
    def forward(self, x):
        return splitB(x, 3)
print(torch.jit.script(NetB())) # fail