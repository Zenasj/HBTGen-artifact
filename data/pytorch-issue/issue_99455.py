import torch

def foo():
    x = torch.randn(5, 5, requires_grad=True)
    y = x + 2
    return x, y

x, y = foo()
print(x.requires_grad, y.requires_grad)
# True True

x, y = torch.compile()(foo)()
print(x.requires_grad, y.requires_grad)
# False False