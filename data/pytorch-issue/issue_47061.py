import torch

x = torch.tensor([1.0], requires_grad=True);
grad = torch.autograd.grad(2*x, [], grad_outputs=torch.tensor([1.0]))
assert(x.grad is None)