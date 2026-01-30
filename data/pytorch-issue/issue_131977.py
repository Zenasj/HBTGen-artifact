py
import torch

ind = torch.tensor([[1], [0], [0]], device="cuda")
val = torch.tensor([1.], device="cuda")
A = torch.sparse_coo_tensor(ind, val, size=(2, 1, 1))
B = torch.zeros((2, 1, 1), device="cuda")
C = torch.bmm(A, B)