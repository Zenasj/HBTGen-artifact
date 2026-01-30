import torch

In [10]: k = torch.rand([0, 3])  # empty tensor; tensor([], size=(0, 3))

In [17]: torch.amax(k)
Out[17]: tensor(-5.9405e-18)

In [18]: torch.amin(k)
Out[18]: tensor(-1.2214e+14)