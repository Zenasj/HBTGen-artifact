import torch

x = torch.rand(5, 5, device='cpu')

def f(x):

    # All of the transforms raise : 
    # torch._dynamo.exc.InternalTorchDynamoError: argument of type: <class 'builtin_function_or_method'>
    
    # vmap
    # return torch.func.vmap(torch.sum)(x)

    # grad
    # return torch.func.grad(torch.sum)(x)

    # o, fn = torch.func.vjp(torch.sum, *(x,))
    # return fn(torch.randn_like(o))

    # return torch.func.jvp(torch.sum, (x,), (torch.randn_like(x),))

    # return torch.func.jacrev(torch.sum)(x)
    # return torch.func.jacfwd(torch.sum)(x)

    return x

a = f(x)
e = torch.compile(f)(x)

torch.testing.assert_close(a, e)

from torch.func import vmap, jacrev, grad
import torch
import time

a = torch.rand(10000, 10000)


def f(x):
    return torch.sin(x).sum(-1)


def df(x):
    return torch.cos(x)


if __name__ == '__main__':
    t0 = time.time()
    grad_f = torch.compile(grad(f))
    t1 = time.time()
    print("compile time:", t1 - t0)

    t0 = time.time()
    b = df(a)
    t1 = time.time()
    c = vmap(grad_f)(a)
    t2 = time.time()
    print(t1 - t0, t2 - t1)

    assert torch.allclose(b, c)

# ...
a = torch.rand(10000)
# ...
if __name__ == '__main__':
    # ...
    c = grad_f(a)
    # ...

import torch

from torch.utils import benchmark

def df(x):
    return torch.cos(x)

# def df(x):
#     return torch.cos(x) + torch.sin(x) + torch.cos(x) + torch.sin(x)

result = []
for shape in ((1,), (1000,), (1000, 1000)):

    x = torch.randn(shape)
    actual = df(x)
    
    compiled_fn = torch.compile(df)
    expected = compiled_fn(x)

    # Sanity check
    torch.testing.assert_close(actual, expected)

    result.append(benchmark.Timer(
        stmt='df(x)',
        globals={'df': df, 'x': x},
        label='Timings',
        sub_label=f'{shape}',
        description=f'non-compiled',
    ).blocked_autorange(min_run_time=1))

    result.append(benchmark.Timer(
        stmt='df(x)',
        globals={'df': compiled_fn, 'x': x},
        label='Timings',
        sub_label=f'{shape}',
        description=f'compiled',
    ).blocked_autorange(min_run_time=1))

print(benchmark.Compare(result))

def df(x):
    # In compilation, it doesn't recompute `cos` and `sin`
    # x = cos(x)
    # y = sin(x)
    # return x + x + y + y
    return torch.cos(x) + torch.sin(x) + torch.cos(x) + torch.sin(x)