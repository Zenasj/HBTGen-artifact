import torch

In [23]: torch.tensor(12, dtype=torch.int8) * torch.tensor(12, dtype=torch.int8)
Out[23]: tensor(-112, dtype=torch.int8)
In [24]: 12 * 12
Out[24]: 144
In [25]: torch.iinfo(torch.int8)
Out[25]: iinfo(min=-128, max=127, dtype=int8)