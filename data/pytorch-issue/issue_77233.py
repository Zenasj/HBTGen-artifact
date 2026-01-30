import torch

a = torch.rand([3, 2, 1], dtype=torch.float64)
b = torch.rand([1, 2, 0], dtype=torch.complex128)
torch.tensordot(a, b)
# tensor([], size=(3, 0), dtype=torch.float64)

import torch

a = torch.rand([3, 2, 1], dtype=torch.float64)
b = torch.rand([1, 2, 1], dtype=torch.complex128)
torch.tensordot(a, b)
# RuntimeError: expected scalar type Double but found ComplexDouble

import torch

a = torch.rand([2, 1], dtype=torch.float64)
b = torch.rand([1, 0], dtype=torch.complex128)
torch.tensordot(a, b)
# RuntimeError: expected scalar type Double but found ComplexDouble