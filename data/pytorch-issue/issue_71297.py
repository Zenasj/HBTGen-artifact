import torch

a = torch._sparse_csr_tensor_unsafe(
        crow_indices=torch.tensor([0, 1]),
        col_indices=torch.tensor([0]),
        values=torch.tensor([[[1, 2], [3, 4]]]),
        device='cuda:0',
        size=(2, 2), dtype=torch.complex128)

b = torch.tensor([[1], [2]], device='cuda:0', dtype=torch.complex128)


torch.cuda.synchronize()    # before
torch.triangular_solve(b, a, upper=True, unitriangular=True, transpose=True)
torch.cuda.synchronize()    # after