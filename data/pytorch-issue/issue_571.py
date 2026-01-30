import torch
import numpy as np
import random

In [3]: np.random.rand(5)[10:100]
Out[3]: array([], dtype=float64)

In [5]: a = torch.rand(5, 5)

In [6]: a[:2, 10:100]
Out[6]: tensor([])

In [7]: torch.__version__
Out[7]: '0.5.0a0+0515664'

In [20]: q = torch.rand(2, 128).cuda()                           

In [21]: q[[2]].size()          
Out[21]: torch.Size([1, 128])