import torch
import torch.nn as nn

In [20]: mp = torch.nn.MaxPool1d(2)

In [21]: x = torch.rand(32, 768)

In [22]: mp(x).shape
Out[22]: torch.Size([32, 384])

In [23]: f = torch.nn.Linear(768, 768)

In [24]: y = f(x)

In [25]: y.shape
Out[25]: torch.Size([32, 768])

In [26]: mp(y).shape
Out[26]: torch.Size([32, 384])