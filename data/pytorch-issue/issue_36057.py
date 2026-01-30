import torch
a = torch.tensor((True,), dtype=torch.bool)
b = torch.tensor((1 + 1j,), dtype=torch.complex128)
print(a + b)