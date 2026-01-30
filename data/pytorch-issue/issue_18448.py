import torch

In [15]: x = torch.eye(512).div(500000)

In [16]: x.slogdet()
Out[16]: (tensor(0.), tensor(-6718.6489))

In [17]: x.logdet()
Out[17]: tensor(-inf)

In [12]: x = torch.eye(512).div(500000)

In [13]: x.slogdet()
Out[13]: (tensor(1.), tensor(-6718.6489))

In [14]: x.logdet()
Out[14]: tensor(-6718.6489)