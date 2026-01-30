import torch

def some_fn(x):
    return torch.sin(x) + 20

def foo(x, y):
    x = some_fn(x)

    def add(x, y):
        return x + y

    return add(x, y)

x = torch.ones(1,)
y = torch.ones(1,)

opt = torch.compile(foo, fullgraph=True, backend='aot_eager')
print("ACTUAL:", opt(x, y), " EXPECTED:", foo(x, y))

# Change some_fn
some_fn = lambda x: torch.cos(x)

print("ACTUAL:", opt(x, y), " EXPECTED:", foo(x, y))