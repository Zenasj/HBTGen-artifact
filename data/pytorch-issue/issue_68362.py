import torch

x = torch.randn(5, requires_grad=True)
y = x.pow(2)
print(x.equal(y.grad_fn._saved_self))  # AttributeError: 'PowBackward0' object has no attribute '_saved_self'
print(x is y.grad_fn._saved_self)  # Similar AttributeError