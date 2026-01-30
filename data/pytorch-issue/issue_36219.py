import torch

y, x = torch.meshgrid(torch.arange(2), torch.arange(3))
assert (not y.is_contiguous()) and (not x.is_contiguous())

print(x)
print(y)
"""
tensor([[0, 1, 2],
        [0, 1, 2]])
tensor([[0, 0, 0],
        [1, 1, 1]])
"""

dx = torch.ones(2, 3)
dy = torch.ones(2, 3)

print(x + dx)
print(y + dy)
"""
tensor([[1., 2., 3.],
        [1., 2., 3.]])
tensor([[1., 1., 1.],
        [1., 1., 1.]]) <- ERROR!
"""

print(x.contiguous() + dx)
print(y.contiguous() + dy)
# the output is correct here