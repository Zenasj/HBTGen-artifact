import torch._dynamo
import logging

def g(a):
    b = a * 2
    c = a * 2
    return b, c

x = torch.rand((1000000,), device="cuda", requires_grad=True)
expect = g(x)
actual = torch._dynamo.optimize("inductor")(g)(x)
assert expect[0] is not expect[1]
assert actual[0] is actual[1]