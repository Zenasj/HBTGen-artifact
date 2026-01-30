import torch
A = torch.rand([0, 4, 4, 3, 0], dtype=torch.float64)
B = torch.rand([0, 3, 3], dtype=torch.float64)
print(torch.linalg.tensorsolve(A, B))
# tensor([], size=(3, 0), dtype=torch.float64)
x = torch.linalg.tensorinv(A)
torch.tensordot(x, B)
# RuntimeError: contracted dimensions need to match, but first has size 4 in dim -1 and second has size 3 in dim 1

import torch
A= torch.rand([6, 4, 4, 3, 2], dtype=torch.float64)
B= torch.rand([6, 4, 2], dtype=torch.float64)

x = torch.linalg.tensorinv(A)
print(torch.tensordot(x, B))
# succeed

print(torch.linalg.tensorsolve(A, B))
# RuntimeError: Expected self to satisfy the requirement prod(self.shape[other.ndim:]) == prod(self.shape[:other.ndim]), but got 6 != 48