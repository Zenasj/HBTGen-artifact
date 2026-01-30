import torch
from scipy import special

x = torch.rand((100,))
a = torch.rand((100,)) * 10
b = torch.rand((100,)) * 10

from_torch = torch.special.betainc(x, a, b)
from_scipy = special.betainc(a, b, x)

assert torch.isclose(from_torch, from_scipy.float()).all() # works!

import torch
from scipy import special

x = torch.rand((100,))
a = torch.rand((100,)) * 10
b = torch.rand((100,)) * 10

from_torch = torch.special.betainc(x, a, b)
from_scipy = special.betainc(a, b, x)

assert torch.isclose(from_torch, from_scipy.float()).all() # works!