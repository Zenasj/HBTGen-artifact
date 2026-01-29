# torch.rand(B, 10, dtype=torch.double)
import torch
import torch.nn as nn

class LU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        lu, pivots = A.lu()
        p, l, u = torch.lu_unpack(lu, pivots)
        ctx.save_for_backward(l, u, p)
        ctx.mark_non_differentiable(pivots)
        return l, u, p

    @staticmethod
    def backward(ctx, L_grad, U_grad, P_grad):
        L, U, P = ctx.saved_tensors
        Ltinv = L.inverse().transpose(-1, -2)
        Utinv = U.inverse().transpose(-1, -2)
        phi_l = (L.transpose(-1, -2) @ L_grad).tril_()
        phi_l.diagonal(dim1=-2, dim2=-1).mul_(0.0)
        phi_u = (U_grad @ U.transpose(-1, -2)).triu_()
        A_grad = Ltinv @ (phi_l + phi_u) @ Utinv
        return P @ A_grad

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.A = nn.Parameter(torch.randn(10, 10, dtype=torch.double, requires_grad=True))

    def forward(self, x):
        L, U, P = LU.apply(self.A)
        b_permuted = P.t() @ x.unsqueeze(-1)
        y = torch.triangular_solve(b_permuted, L, upper=False).solution
        x_sol = torch.triangular_solve(y, U, upper=True).solution
        return x_sol.squeeze(-1)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 10, dtype=torch.double)

