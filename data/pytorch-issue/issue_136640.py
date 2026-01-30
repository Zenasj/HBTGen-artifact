import torch

@torch.compile(dynamic=True)
def f(x, y):
    x_view = x.view(-1, 4)
    y_view = y.view(-1, 4)
    return x_view * y_view


x = torch.randn(4)
y = torch.randn(8)
out = f(x, y)