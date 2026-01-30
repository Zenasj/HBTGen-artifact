import torch

def foo(x, y):
    z = x * y
    for i in range(10):
        z = z * y
        print(z)
    return z


a = torch.randn([2, 2])
b = torch.randn([2, 2])

foo = torch._dynamo.optimize('eager')(foo)

foo(a, b)