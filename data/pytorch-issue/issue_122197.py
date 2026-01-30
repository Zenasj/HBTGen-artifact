import torch
torch._dynamo.config.capture_func_transforms = True
jvp = torch.func.jvp


def f(x, y, z):
    a, b = x
    return a + 2 * b + 3 * y + 4 * z

@torch.compile(backend='aot_eager', fullgraph=True)
def fn(x):
    return jvp(f, ((x, x,), x, x), ((x, x), x, x))


x = torch.tensor(1.,)
y = fn(x)
print(y)