import torch
import torch.nn as nn

In [6]: d = torch.nn.Dropout(p=0.1)

In [7]: x = torch.full([1000], float("Inf")).cuda()

In [8]: torch.isnan(d(x)).any().item()
Out[8]: True