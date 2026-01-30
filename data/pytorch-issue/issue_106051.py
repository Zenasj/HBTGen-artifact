import torch

x0 = torch.tensor([5 + 1j, 2 + 2j])
x0 = torch.conj(x0)
print(x0); print()

x1 = torch.tensor([5 + 1j, 2 + 2j])
xc1 = torch.conj(x1)
print(xc1)
x1[:] = xc1
print(x1); print()

x2 = torch.tensor([5 + 1j, 2 + 2j])
xc2 = torch.tensor([5 - 1j, 2 - 2j])
print(xc2)
x2[:] = xc2
print(x2); print()

print(xc1 == xc2)
print(xc1.storage())
print(xc2.storage())

x = torch.tensor([1 + 1j])
xc = torch.conj(x)
xc[:] = torch.resolve_conj(xc)
print(xc)
print(x)

x = torch.tensor([1 + 1j])
xc = torch.conj(x)
xc[:] = torch.conj_physical(xc)
print(xc)
print(x)