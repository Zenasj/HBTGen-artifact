import torch

nt0 = torch.nested.nested_tensor([torch.rand(2, 6), torch.rand(3, 6)], layout=torch.jagged, requires_grad=True)
nt1 = torch.nested.nested_tensor_from_jagged(torch.rand(5, 6), offsets=nt0.offsets()).requires_grad_(True)

out = torch.matmul(nt0.transpose(-2, -1), nt1)
out.sum().backward()