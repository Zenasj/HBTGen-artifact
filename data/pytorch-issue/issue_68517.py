import torch

a = torch.rand(1)
b = torch.rand(1)

E1 = a**2-2*a*b+b**2
E2 = torch.square(a-b)

print(torch.isclose(E1, E2))