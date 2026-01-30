import random

import torch
a = torch.randn(3,4,5)
b = torch.sum(a, dim=())
assert a.shape == b.shape   # raises assertion error

import numpy
a = numpy.random.randn(3,4,5)
b = numpy.sum(a, axis=())
assert a.shape == b.shape   # passes