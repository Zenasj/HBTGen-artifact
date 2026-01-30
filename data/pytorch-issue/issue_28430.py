import torch

In [32]: a = torch.ones(3); b = torch.ones(3) * 2; c = torch.ones(3) * 3

# do not pass optional "value"
In [38]: torch.addcdiv(a, b, c)
Out[38]: tensor([1.6667, 1.6667, 1.6667])

# pass optional "value", but by position
In [39]: torch.addcdiv(a, 3, b, c)
Out[39]: tensor([3., 3., 3.])

# can also pass "value" by keyword
In [40]: torch.addcdiv(a, b, c, value=3)
Out[40]: tensor([3., 3., 3.])