import torch

a = torch.rand((2, 2))  # row-major
b = a.T  # column-major

print(a.stride())
# output: (2, 1), expected

print(b.stride())
# output: (1, 2), expected

print(b.to(memory_format=torch.contiguous_format).stride())
# output: (1, 2), unexpected, shouldn't it be transformed to row-major?

print(b.contiguous().stride())
# output: (2, 1), this works fine