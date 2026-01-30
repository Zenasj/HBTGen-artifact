import torch

def hook(t):
    t.grad.mul_(5)

x = torch.ones([2, 2], requires_grad=True)
y = torch.ones([2, 2], requires_grad=True)
x.register_post_accumulate_grad_hook(hook)

def fn(x, y):
    return x + y

out = fn(x, y)
x.backward(torch.ones([2, 2]))
print(x.grad)

with compiled_autograd.enable(torch.compile(backend="inductor", fullgraph=True)):
    out = fn(x, y)
    x.backward(torch.ones([2, 2]))

out = fn(x, y)
x.backward(torch.ones([2, 2]))
print(x.grad)