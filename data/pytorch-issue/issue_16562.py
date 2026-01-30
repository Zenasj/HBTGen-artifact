import torch
a = torch.rand(2, 3, 4)
b = torch.rand(4, 6)
b_ = b.to_sparse()

c = torch.matmul(a, b)
# torch.Size([2, 3, 6])

c_ = torch.matmul(b.t(), a.flatten(end_dim = 1).t()).view(-1, *a.shape[:2]).permute(1, 2, 0)
print(torch.eq(c, c_).all()) # 1

#print(torch.matmul(a, b_).shape)
# RuntimeError: Expected object of backend CPU but got backend SparseCPU for argument #2 'mat2'

import torch
a = torch.rand(2, 3, 4)
b = torch.rand(4, 6)

print(b.t() @ a.permute(2, 0, 1))
# Correct error about shape:     RuntimeError: Expected tensor to have size 4 at dimension 1, but got size 2 for argument #2 'batch2' (while checking arguments for bmm)

print(b.to_sparse().t() @ a.permute(2, 0, 1))
# Different and cryptic error: RuntimeError: sparse tensors do not have strides