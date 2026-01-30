import torch
import numpy as np

a = torch.arange(4).reshape(2,2)
a += a.t()
# tensor([[0, 3],
#         [5, 6]])

b = torch.arange(4).reshape(2,2)
b = b + b.t()
# tensor([[0, 3],
#         [3, 6]])

# Numpy (1.19.1) produces consistent output:

c = np.arange(4).reshape(2,2)
c += c.T
# array([[0, 3],
#        [3, 6]])

d = np.arange(4).reshape(2,2)
d = d + d.T
# array([[0, 3],
#        [3, 6]])