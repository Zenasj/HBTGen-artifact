import torch

class C:
    x = 1

    def __str__(self):
        return "ok"

def foo(x):
    a = C()
    return x, str(a)

print(torch.compile(foo, fullgraph=True)(torch.ones(4)))