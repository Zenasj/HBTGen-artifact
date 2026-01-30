import torch

x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
ord = 3+4j
norm = torch._C._linalg.linalg_vector_norm(x, ord=ord)