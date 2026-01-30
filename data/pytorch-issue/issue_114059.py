import torch

# We clearly have 2 graphs
def fn(x):
    x = x + 2
    torch._dynamo.graph_break()
    return x + 1

opt_fn = torch._dynamo.optimize(backend="eager")(fn)
nopython_fn = torch._dynamo.optimize(backend="eager", nopython=True)(fn)

try:
    nopython_fn(torch.zeros(1))
except Exception:
    print("failed to run")

print(opt_fn(torch.zeros(1)))
print(nopython_fn(torch.zeros(1)))

def fn(x):
    x = x + 2
    torch._dynamo.graph_break()
    return x + 1

opt_fn = torch.compile(fn, backend="eager")
nopython_fn = torch.compile(fn, backend="eager", fullgraph=True)

try:
    nopython_fn(torch.zeros(1))
except Exception:
    print("failed to run")

print(opt_fn(torch.zeros(1)))
print(nopython_fn(torch.zeros(1)))

def fn(x):
    x = x + 2
    torch._dynamo.graph_break()
    return x + 1

opt_fn = torch._dynamo.optimize(backend="eager")(fn)
nopython_fn = torch.compile(fn, backend="eager", fullgraph=True)

try:
    nopython_fn(torch.zeros(1))
except Exception:
    print("failed to run")

print(opt_fn(torch.zeros(1)))
print(nopython_fn(torch.zeros(1)))