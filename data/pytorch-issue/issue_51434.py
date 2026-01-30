a, b = torch.nonzero(torch.randn(10, 10), as_tuple=True)

import torch

t = torch.randn(10, 10)
reveal_type(t)

# Annotated as Tensor, but gives Any rather than complaining
a, b = torch.nonzero(t, as_tuple=True)
reveal_type(a)
reveal_type(b)

c = torch.nonzero(t, as_tuple=True)
reveal_type(c)

d = torch.nonzero(t, as_tuple=False)
reveal_type(d)