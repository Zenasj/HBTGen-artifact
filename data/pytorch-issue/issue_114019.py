import torch

class MyCallable:
    def __call__(self, x):
        return x + 1

def fn(x):
    return MyCallable()(x)

fn_opt = torch.compile(fn, backend="eager")
print(fn_opt(torch.zeros(1)))