import torch

a = torch.rand(10, requires_grad=True)

b = a * 2

print(b.grad_fn._saved_other) # Works fine

b.sum().backward()

print(b.grad_fn._saved_other) # Hard crash