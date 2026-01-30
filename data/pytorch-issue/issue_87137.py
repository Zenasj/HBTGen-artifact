import torch
import numpy as np

A = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
A = A.div(A.sum(1, keepdim=True))
B = A.clone().cuda()
b = torch.zeros(3, 3, dtype=torch.float64, requires_grad=True)
c = b.clone().cuda()
x = torch.lstsq(A, b)[0]
y = torch.lstsq(B, c)[0]
print(x)  # on cpu it gives [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
print(y)  # on gpu it gives [[nan, nan, nan], [nan, nan, nan], [inf, -inf, inf]]

print(np.linalg.lstsq(A.detach().numpy(), x.detach().numpy()))  # numpy gives [[0, 0, 0], [0, 0, 0], [0, 0, 0]]