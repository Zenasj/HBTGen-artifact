import torch

# batch matrix multiplication; with broadcasting
As = torch.randn(1, 2, 5)
Bs = torch.randn(3, 5, 4)
As @ BS                               # this works
torch.einsum('bij,bjk->bik', As, Bs)  # this does not work

import numpy

As = numpy.zeros(shape=(1, 2, 5))
Bs = numpy.zeros(shape=(3, 5, 4))
numpy.einsum('bij,bjk->bik', As, Bs)