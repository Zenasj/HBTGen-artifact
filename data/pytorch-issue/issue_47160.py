import torch, numpy
a=numpy.arange(5.)
a.flags.writeable=False
t=torch.tensor(a)