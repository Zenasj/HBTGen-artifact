import torch
import torch.nn as nn

foo = torch.arange(25).reshape((1, 5, 5))
f = nn.functional.pad(foo, (2, 2, 2, 2), mode='circular')

f = nn.functional.pad(foo, (2, 2), mode='circular')
f = torch.moveaxis(nn.functional.pad(torch.moveaxis(f, -1, -2), (2, 2), mode='circular'), -1, -2)
f

"""
tensor([[[18, 19, 15, 16, 17, 18, 19, 15, 16],
         [23, 24, 20, 21, 22, 23, 24, 20, 21],
         [ 3,  4,  0,  1,  2,  3,  4,  0,  1],
         [ 8,  9,  5,  6,  7,  8,  9,  5,  6],
         [13, 14, 10, 11, 12, 13, 14, 10, 11],
         [18, 19, 15, 16, 17, 18, 19, 15, 16],
         [23, 24, 20, 21, 22, 23, 24, 20, 21],
         [ 3,  4,  0,  1,  2,  3,  4,  0,  1],
         [ 8,  9,  5,  6,  7,  8,  9,  5,  6]]])
"""