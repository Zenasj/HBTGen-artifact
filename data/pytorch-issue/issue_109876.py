import torch

x = torch.randn(5, 5, requires_grad=True)

y = torch.mean(x)
z = torch.sum(x)

print(y, z)
print(list(filter(lambda x: x.startswith('_saved_'), dir(y.grad_fn))), list(filter(lambda x: x.startswith('_saved_'), dir(z.grad_fn))))
print(y.grad_fn._saved_self is x)