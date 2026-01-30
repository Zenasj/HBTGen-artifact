import torch

x = torch.randn(3, 3, requires_grad=True)
y = x.clone()
gy = torch.randn(3, 3).t()
res1, = torch.autograd.grad(y, x, gy)

x = torch.randn(3, 3, requires_grad=True)
y = x.clone()
gy = torch.randn(3, 3).t()
y.backward(gy)
res2 = x.grad

print(res1.stride())
(1, 3)
print(res2.stride())
(3, 1)