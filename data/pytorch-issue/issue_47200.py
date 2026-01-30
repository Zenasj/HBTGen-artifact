python
import torch
print(torch.__version__)
# 1.8.0.dev20201101

z = torch.ones(1, requires_grad=True, dtype=torch.complex128)
out = 0.5 * (z - z.conj())
out.backward()
print(z.grad)
# tensor([0.+0.j], dtype=torch.complex128)

z = torch.ones(1, requires_grad=True, dtype=torch.complex128)
out = z.imag
out.backward()
print(z.grad)
# tensor([0.+1.j], dtype=torch.complex128)