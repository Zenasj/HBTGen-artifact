import torch

@torch.compile()
def foo(x, y):
    code = compile(y, "foo", "exec")
    exec(y)
    return x

print(foo(torch.rand(3), "print('Hello World')"))