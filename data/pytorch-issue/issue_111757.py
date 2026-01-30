import torch
x = torch.ones((1, 0))
y = torch.ones((1,))
z = torch.ones((1,))
# works; b is just a zero-sized axis as expected
print(torch.einsum('ab,a->b', x, y))

x = torch.ones((1, 1, 0))
y = torch.ones((1,))
z = torch.ones((1,))
# no similar luck with an extra argument in the mix; we get an error
#     self.speedup = self.naive_cost / self.opt_cost
#     decimal.InvalidOperation: [<class 'decimal.DivisionUndefined'>]
print(torch.einsum('abc,a,b->c', x, y, z))

# numpy and JAX are a-ok with these scenarios, as expected
import numpy as np
print(np.einsum('abc,a,b->c', x, y, z))