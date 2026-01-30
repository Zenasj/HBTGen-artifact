import torch

In [2]: x = torch.randn((4, 3, 8, 8 ), device='cpu').contiguous(memory_format=torch.channels_last)
In [3]: r1=x.sqrt()
In [4]: r2=x.contiguous().sqrt()
In [5]: diff=torch.abs(r1-r2)
In [6]: diff.max()
Out[6]: tensor(nan)
In [7]: diff[diff==diff].max()
Out[7]: tensor(0.)
In [8]: diff[ (diff!=0) & (diff==diff) ]
Out[8]: tensor([])
In [9]: (diff/1).max()
Out[9]: tensor(5.3258e+34)