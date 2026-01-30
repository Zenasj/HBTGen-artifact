import torch

def hook0(_unused_grad):
    print("hook0")

def hook1(_unused_grad):
    print("hook1")

a0 = torch.tensor(3., requires_grad=True)
a1 = torch.tensor(5., requires_grad=True)
a = [a0, a1]
b0, b1 = torch._foreach_add(a, 1)
c0 = b0.clone()
c0.register_hook(hook0)
c1 = b1.clone()
c1.register_hook(hook1)
torch.autograd.grad(c0 * c1, inputs=(a0, a1), retain_graph=True)  # expect both hooks fired
torch.autograd.grad(c0 * c1, inputs=(a0,), retain_graph=True) # expect only hook0 fired
torch.autograd.grad(c0 * c1, inputs=(a1,), retain_graph=True) # expect only hook1 fired