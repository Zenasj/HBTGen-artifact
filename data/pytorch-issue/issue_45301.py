import torch
x = torch.rand(2, 2, requires_grad=True)
with torch.no_grad():
    y = x[:, :1]

z = y.clone()
print(y.requires_grad)  # True
print(z.grad_fn.next_functions)  # ((None, 0),)