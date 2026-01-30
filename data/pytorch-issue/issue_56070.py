import torch
import torch.nn as nn

class Model (torch.nn.Module):
    def forward(self, x):
        y = x.view(-1)
        z = torch.tensor(2.0).float()
        y.add_(z)
        return x

m = Model()
x = torch.rand(2, 3)
y = m(x)

class Model (torch.nn.Module):
    def forward(self, x):
        y = x.transpose(1, 2)
        z = torch.tensor(2.0).float()
        x.add_(z)
        return y

m = Model()
x = torch.rand(1, 2, 3)
y = m(x)