import torch

input = torch.rand([10, 3, 5], dtype=torch.float64, requires_grad=True)
batch1 = torch.rand([10, 3, 4], dtype=torch.complex128, requires_grad=True)
batch2 = torch.rand([10, 4, 5], dtype=torch.complex128, requires_grad=True)

res = torch.baddbmm(input, batch1, batch2)
res.sum().abs().backward()

input = torch.rand([5, 3], dtype=torch.complex128, requires_grad=True)
other = torch.rand([5, 3], dtype=torch.float64, requires_grad=True)

res = torch.dist(input, other)
res.sum().abs().backward()

input = torch.rand([3], dtype=torch.float64, requires_grad=True)
vec1 = torch.rand([3, 2], dtype=torch.complex128, requires_grad=True)
vec2 = torch.rand([2], dtype=torch.complex128, requires_grad=True)

res = torch.Tensor.addmv(input, vec1, vec2)
res.sum().abs().backward()

input = torch.rand([3], dtype=torch.complex128, requires_grad=True)
vec1 = torch.rand([3, 2], dtype=torch.float64, requires_grad=True)
vec2 = torch.rand([2], dtype=torch.float64, requires_grad=True)

res = torch.Tensor.addmv(input, vec1, vec2)
desired_dtype = torch.promote_types(torch.complex128, torch.float64)
print(res.dtype)
print(desired_dtype)

a = torch.rand([5, 3], dtype=torch.float64, requires_grad=True)
b = torch.rand([5, 3], dtype=torch.complex128, requires_grad=True)
res = torch.dist(a, b)
desired_dtype = torch.promote_types(torch.complex128, torch.float64)
print(res.dtype)
print(desired_dtype)
# Output
# torch.float64
# torch.complex128

input = torch.rand([10, 3, 5], dtype=torch.complex128, requires_grad=True)
batch1 = torch.rand([10, 3, 4], dtype=torch.float64, requires_grad=True)
batch2 = torch.rand([10, 4, 5], dtype=torch.float64, requires_grad=True)
res = torch.baddbmm(input, batch1, batch2)
desired_dtype = torch.promote_types(torch.complex128, torch.float64)
# res.sum().abs().backward()
print(res.dtype)
print(desired_dtype)
# Output
# torch.float64
# torch.complex128