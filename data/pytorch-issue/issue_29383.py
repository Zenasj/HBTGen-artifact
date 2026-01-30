import torch

x = torch.randn(3, requires_grad=True)
y = x * 2
print(y)

gradients = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(gradients)
print(x.grad)

print("all done")

x = torch.tensor(3., requires_grad=True)
y = x*2
y.backward()