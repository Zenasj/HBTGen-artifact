import torch
import torch.nn as nn
x = torch.randn(1,2048,1)
avpool = nn.AvgPool1d(2)
y = avpool(x)
print(y.shape)