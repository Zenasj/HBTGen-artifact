import torch

def cool_name(x):
    return x.sin()

def fn(x):
    return torch.no_grad(cool_name)(x)

x = torch.zeros(10)
result = fn(x)
print(result)
result = torch.compile(fn, backend="eager", fullgraph=True)(x)
print(result)

def fn(x):
    @torch.no_grad
    def cool_name(x):
        return x.sin()

    return cool_name(x)

x = torch.zeros(10)
result = fn(x)
print(result)
result = torch.compile(fn, backend="eager", fullgraph=True)(x)
print(result)

@torch.no_grad
def cool_name(x):
    return x.sin()

def fn(x):
    return cool_name(x)

x = torch.zeros(10)
result = fn(x)
print(result)
result = torch.compile(fn, backend="eager", fullgraph=True)(x)
print(result)