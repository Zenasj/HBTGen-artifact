import torch

A = torch.rand([], dtype=torch.float64)
ord = torch.rand([], dtype=torch.float64)
torch.linalg.vector_norm(A, ord=ord)
# tensor(0.1132, dtype=torch.float64)

import torch

A = torch.rand([], dtype=torch.float64)
ord = torch.rand([], dtype=torch.float64).requires_grad_()
torch.linalg.vector_norm(A, ord=ord)
# TypeError: linalg_vector_norm(): argument 'ord' must be Number, not Tensor