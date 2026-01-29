# torch.rand(1, 1, 64, 64, dtype=torch.float32)  # Input shape: batch, channels, height, width (matrix dimensions)
import torch
from torch import nn

def compute_grad_V(U, S, V, grad_V):
    N = S.shape[0]
    K = svd_grad_K(S)
    S_diag = torch.eye(N, device=S.device) * S.view((N, 1))
    inner = K.T * (V.T @ grad_V)
    inner = (inner + inner.T) / 2.0
    return 2 * U @ S_diag @ inner @ V.T

def svd_grad_K(S):
    N = S.shape[0]
    s1 = S.view((1, N))
    s2 = S.view((N, 1))
    diff = s2 - s1
    plus = s2 + s1

    eps = torch.ones((N, N), device=S.device) * 1e-6
    max_diff = torch.max(torch.abs(diff), eps)
    sign_diff = torch.sign(diff)

    K_neg = sign_diff * max_diff
    K_neg[torch.arange(N), torch.arange(N)] = 1e-6
    K_neg = 1.0 / K_neg
    K_pos = 1.0 / plus

    ones = torch.ones((N, N), device=S.device)
    rm_diag = ones - torch.eye(N, device=S.device)
    K = K_neg * K_pos * rm_diag
    return K

class CustomSVD(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        U, S, V = torch.svd(input, some=True)
        ctx.save_for_backward(U, S, V)
        return U, S, V

    @staticmethod
    def backward(ctx, grad_U, grad_S, grad_V):
        U, S, V = ctx.saved_tensors
        grad_input = compute_grad_V(U, S, V, grad_V)
        return grad_input

customsvd = CustomSVD.apply

class MyModel(nn.Module):
    def forward(self, x):
        matrix = x.squeeze()  # Convert to 2D matrix (64x64)
        U, S, V = customsvd(matrix)
        return S  # Return singular values as output

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1, 64, 64, dtype=torch.float32).cuda()

