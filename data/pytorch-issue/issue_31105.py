import torch

a1 = torch.rand([4, 4], requires_grad=True)
b1 = a1.squeeze(0)**2
b1.sum().backward()
print(a1.grad)

a2 = torch.rand([1, 4, 4], requires_grad=True)
b2 = a2.unsqueeze(0)**2
b2.sum().backward()
print(a2.grad)