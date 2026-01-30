import numpy as np
import torch
print(torch.__version__)
A = np.array([[0.7944, 0.6003], [0.6003, 0.6311]])
U, S, V = np.linalg.svd(A)
print("numpy: U: ")
print(U)
print(U@np.diag(S)@V)


U, S, V = torch.svd(torch.from_numpy(A))
print("torch_cpu U: ")
print(U)
print(U@torch.diag(S)@V)


U, S, V = torch.svd(torch.from_numpy(A).cuda())
print("torch_cuda U: ")
print(U)
print(U@torch.diag(S)@V)