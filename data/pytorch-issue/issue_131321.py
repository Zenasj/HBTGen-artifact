import torch

src = torch.tensor([1., 2., 3., 4., 5., 6.])
index = torch.tensor([0, 1, 0, 1, 2, -1])
input = torch.tensor([1., 2., 3., 4.])
compiled_sr = torch.compile(torch.scatter_reduce)

# Throws "index out of bounds" error
print(compiled_sr(input, 0, index, src, "sum"))

# Throws "index out of bounds" error
print(torch.scatter_reduce(input, 0, index, src, "sum"))

src = src.cuda()
index = index.cuda()
input = input.cuda()

# [ 5.,  8.,  8., 10.]  BAD !
print(compiled_sr(input, 0, index, src, "sum"))

# Throws "index out of bounds" error
print(torch.scatter_reduce(input, 0, index, src, "sum"))