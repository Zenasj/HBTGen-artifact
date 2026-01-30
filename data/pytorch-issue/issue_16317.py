import torch

x = torch.tensor([0.0, 1.0], requires_grad=True)
n = torch.tensor([0.0, 2.0])
y = x / n
y = torch.where(n==0, torch.zeros_like(y), y)
y.sum().backward()
print(y)
print(x.grad)

x = torch.tensor([0.0, 1.0], requires_grad=True)
n = torch.tensor([0.0, 2.0])
y = x / torch.where(n==0, torch.ones_like(n), n)
y = torch.where(n==0, torch.zeros_like(y), y)
y.sum().backward()
print(y)
print(x.grad)