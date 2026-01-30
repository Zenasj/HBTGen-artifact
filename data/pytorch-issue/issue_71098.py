import torch
input = torch.randint(-1,1,[2], dtype=torch.int64)
other = torch.randint(-1,1,[2, 2], dtype=torch.int64)
torch.eq(input, other)
# succeed
input.eq_(other)
# RuntimeError: output with shape [2] doesn't match the broadcast shape [2, 2]

import torch
input = torch.randint(-2,1,[1, 3], dtype=torch.int64)
mat1 = torch.randint(-2,4,[2, 3], dtype=torch.int64)
mat2 = torch.randint(-8,1,[3, 3], dtype=torch.int64)
torch.addmm(input, mat1, mat2)
# succeed
input.addmm_(mat1, mat2)
# RuntimeError: The input tensor must be a matrix with size 2x3, but got a 2-D tensor with size 1x3