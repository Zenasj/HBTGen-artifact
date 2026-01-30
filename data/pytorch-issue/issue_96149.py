import torch.nn as nn

import torch
import torch._dynamo
from torch import nn
from torchvision import models


class RegNet1(nn.Module):

    def __init__(self):
        super().__init__()
        model = models.regnet_y_400mf()
        modules = list(model.children())
        self.model = nn.Sequential(modules[0], *modules[1][:2])

    def forward(self, x):
        return self.model(x)

m = RegNet1()
x = torch.rand([4, 3, 64, 64])
opt_m = torch._dynamo.optimize("eager")(m)
print(m(x))
print(opt_m(x))