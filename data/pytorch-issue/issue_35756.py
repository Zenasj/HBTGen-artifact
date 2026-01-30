import torch

def func():
  for i in range(N):
    for j in range(N):
      B[i][j] = torch.dot(A[i], A[j])

func()

# successful

N = 100
A = torch.rand(N, 256)
B = A.matmul(A.T)

B.shape
# N x N