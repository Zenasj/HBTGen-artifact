import torch

batch_size = 2
A = torch.eye(256, device="mps")[None, :, :].expand(batch_size, -1, -1) + 0.1 * torch.randn((batch_size, 256, 256), device="mps")
A_cpu = A.cpu()
LU_cpu, pivots_cpu = torch.linalg.lu_factor(A_cpu)
LU, pivots = torch.linalg.lu_factor(A)
torch.testing.assert_close(LU.cpu(), LU_cpu)