import torch

a = torch.rand(2, requires_grad=True)
b = a.clone()
print(b)

torch.save(b, "foo.pth")
c = torch.load("foo.pth")
print(c)