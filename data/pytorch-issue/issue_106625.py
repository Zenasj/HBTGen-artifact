import torch

@torch._dynamo.debug
def inner_foo(x):
    return x * x

def fn(x, y):
    x2 = inner_foo(x)
    return x2 * y


x = torch.rand([4, 10])
y = torch.rand([4, 10])

torch._dynamo.optimize("eager")(fn)(x, y)