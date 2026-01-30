import torch
A = torch.randn(5, 5, device=device, requires_grad=True)  # 5×5
t = torch.rand(7, 3, device=device)                       # 7×3
At = torch.einsum("ij, ...s -> ...sij", A, t)             # 7×3×5×5
eAt = torch.linalg.matrix_exp(At)                         # 7×3×5×5
s = torch.linalg.norm(At)                                 # 1
s.backward()