import torch
import torch.nn as nn

3
In [1]: circular = nn.Conv2d(6, 1, (3, 3), padding=(0, 1), padding_mode='circular')

In [2]: circular(torch.zeros(1, 6, 20, 10)).shape
Out[2]: torch.Size([1, 1, 20, 8])

In [3]: normal = nn.Conv2d(6, 1, (3, 3), padding=(0, 1))

In [4]: normal(torch.zeros(1, 6, 20, 10)).shape
Out[4]: torch.Size([1, 1, 18, 10])