import torch

def pack_hook(x):
    return (x.device, x.cpu())

def unpack_hook(packed):
    device, tensor = packed
    return tensor.to(device)

def f(a):
    return a * a

# Works if opt_f = f
# Error if opt_f = torch.compile(f)
opt_f = torch.compile(f) 

x = torch.randn(5, requires_grad=True)
with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    y = opt_f(x)
y.sum().backward()

print(torch.allclose(x.grad, (2 * x))) # Expect: True

[tasklist]
### Tasks