import torch
A = torch.rand([2, 3, 3], dtype=torch.float64)
B = torch.rand([2, 3], dtype=torch.float64)
print(torch.linalg.solve(A, B))
# succeed
torch.linalg.inv(A) @ B
# RuntimeError: mat1 and mat2 shapes cannot be multiplied (6x3 and 2x3)