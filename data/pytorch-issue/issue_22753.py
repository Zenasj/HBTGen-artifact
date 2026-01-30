import numpy
x = numpy.zeros((5, 0))
y = numpy.repeat(x, repeats=3, axis=1)
assert (y.shape == (5, 0))

import torch
x = torch.zeros((5, 0))
y = torch.repeat_interleave(x, repeats=3, dim=0)
assert (y.shape == (15, 0))