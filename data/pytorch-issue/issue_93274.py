import torch

def fn(input):
    return torch.histogramdd(input, [1, 1, 55, 31, 34])

x = torch.rand([5, 6], dtype=torch.float32)
ret_eager = fn(x)
print('==== Eager mode OK! ====')

compiled = torch.compile(fn)
ret_compiled = compiled(x)
print('==== torchcomp mode OK! ====')

import torch
import numpy

# invalid bins
bins = [1, 1, 1, 1, 1]

# Valid bins, all asserts pass.
# bins = [1, 1, 1, 1, 1, 1]

def fn(input):
    return torch.histogramdd(input, bins)

x = torch.rand([5, 6], dtype=torch.float32)

# ValueError: The dimension of bins must be equal to the dimension of the  sample x.
o1 = numpy.histogramdd(x.numpy(), bins)
o2 = numpy.histogramdd(x.numpy(), bins)
for o_, oo_ in zip(o1, o2):
    numpy.testing.assert_allclose(o_, oo_)

# For invalid bins, consecutive call return incorrect output (with different shapes)
o1 = fn(x)
o2 = fn(x)
for o_, oo_ in zip(o1, o2):
    # AssertionError: The values for attribute 'shape' do not match: torch.Size([1, 1, 1, 1, 1, 273]) != torch.Size([1, 1, 1, 1, 1, 33]).
    torch.testing.assert_close(o_, oo_)