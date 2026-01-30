import torch

def pack_hook(x):
    return x

def unpack_hook(x):
    print("unpacking")
    return x

a = torch.ones(5, requires_grad=True)
b = torch.ones(5, requires_grad=True) * 2

def f(a, b):
    return a * b

opt_f = torch.compile(backend="aot_eager")(f)

print("# eager")
with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    y = f(a, b)

print("# compile")
with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    y = opt_f(a, b)