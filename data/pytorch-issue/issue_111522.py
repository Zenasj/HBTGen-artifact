import torch

class MyCallable:
    def __call__(self, x):
        return x + 1

def fn(x):
    return MyCallable()(x)

fn_opt = torch.compile(fn, backend="eager")
print(fn_opt(torch.zeros(1)))

def patch__call__(obj, x):
    return x + 2

MyCallable.__call__ = patch__call__
print(fn_opt(torch.zeros(1)))  # Incorrectly 1, should be 2