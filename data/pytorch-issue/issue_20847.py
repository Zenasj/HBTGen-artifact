import torch

@torch.jit.script
def fn(x):
    # type: (int) -> int
    if x > 2:
        raise RuntimeError("bad input, {} is too high".format(x))
    return x + 3

print(fn.graph)
print(fn(4))

@torch.jit.script
def foo():
    return tuple([1,2])
foo()