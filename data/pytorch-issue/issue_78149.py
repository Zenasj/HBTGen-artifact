import torch

dim = -1
sparse_grad = True
input_tensor = torch.rand([10, 10, 5], dtype=torch.float64)
index_tensor = torch.randint(0, 1, [6, 0, 2], dtype=torch.uint8)

input = input_tensor.clone()
index = index_tensor.clone()
res1 = torch.gather(input, dim, index, sparse_grad=sparse_grad, )

input = input_tensor.clone().requires_grad_()
index = index_tensor.clone()
res2 = torch.gather(input, dim, index, sparse_grad=sparse_grad, )
# succeed

res2.sum().backward()
# floating point exception