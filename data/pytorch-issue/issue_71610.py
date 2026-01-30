import torch
input = torch.rand([2, 0, 5, 5], dtype=torch.complex128)
A = torch.rand([1, 5, 5], dtype=torch.complex128)
torch.linalg.solve(A, input)
# RuntimeError: falseINTERNAL ASSERT FAILED at "../aten/src/ATen/native/LinearAlgebraUtils.h":244, please report a bug to PyTorch. linalg_solve: (Batch element 0): Argument 335551511 has illegal value. Most certainly there is a bug in the implementation calling the backend library.