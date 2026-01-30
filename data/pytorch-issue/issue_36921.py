py
import torch

# works for (1, 3, 3)
A = torch.rand(1, 1109, 1109)

# works on the CPU
A = A.cuda()

B = torch.eye(A.shape[1], dtype=A.dtype, device=A.device).expand(A.shape[0], -1, -1)

# works for unbatched data
# A = A[0]
# B = B[0]

LU_data, pivots = torch.lu(A)
torch.lu_solve(B, LU_data, pivots)

py
@torch.jit.script
def lu_solve_batched(
    B: torch.Tensor, LU: torch.Tensor, pivots: torch.Tensor
) -> torch.Tensor:
    """Until https://github.com/pytorch/pytorch/issues/36921 is fixed."""
    if B.dim() == 2:
        B = B.expand(LU.shape[0], -1, -1)
    if B.shape[1] < 1025:
        return torch.lu_solve(B, LU, pivots)
    results = torch.empty_like(B)
    for ibatch in range(int(B.shape[0] / 1024)):
        index = torch.arange(
            1024 * ibatch, min(1024 * (ibatch + 1), B.shape[0]), dtype=torch.long
        )
        results[index] = torch.lu_solve(B[index], LU[index], pivots[index])
    return results

py
import torch

A = torch.rand(1, 1109, 1109) + 10*torch.eye(1109).unsqueeze(0)
B = torch.eye(1109).unsqueeze(0)

A = A.cuda()
B = B.cuda()

LU_data, pivots = torch.lu(A)
solved = torch.lu_solve(B, LU_data, pivots)

(A.bmm(solved) - B).abs().max()  # tensor(1.5497e-06, device='cuda:0')

import torch

A = torch.rand(1, 1, 1) + 10*torch.eye(1).unsqueeze(0)
B = torch.rand(1, 1109).unsqueeze(0)

A = A.cuda()
B = B.cuda()

LU_data, pivots = torch.lu(A)
solved = torch.lu_solve(B, LU_data, pivots)

import torch

A = torch.rand(1, 1109, 1109) + 10*torch.eye(1109).unsqueeze(0)
B = torch.eye(1109).unsqueeze(0)

A = A.double().cuda()
B = B.double().cuda()

LU_data, pivots = torch.lu(A)
solved = torch.lu_solve(B, LU_data, pivots)