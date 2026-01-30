bash
In [6]: torch.__version__                                                                                                                                                                     
Out[6]: '1.3.0a0+de394b6'
In [8]: torch.autograd.Variable(torch.Tensor([float('inf')])).sum()

import torch

s = torch.tensor(float('inf')).sum()
print(s)

s = torch.autograd.Variable(torch.Tensor([float('inf')])).sum()
print(s)

tensor(inf)
tensor(inf)