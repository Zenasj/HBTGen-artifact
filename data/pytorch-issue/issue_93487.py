import torch
import torch._dynamo

torch._dynamo.config.verbose = True

@torch.compile(dynamic=True)
def foo(a):
    return torch.zeros_like(a)

x = torch.randint(0, 1024, size=(100,))
r = foo(x)