# torch.rand(m, n, dtype=torch.float32)  # Add a comment line at the top with the inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, A):
        u, s, v = torch.svd(A)
        
        m, n = A.shape[-2], A.shape[-1]
        k = min(m, n)
        
        ut = u.transpose(-2, -1)
        in_n = torch.eye(n, device=A.device, dtype=A.dtype)
        sigma_mat = s.diag_embed()
        sigma_mat_inv = s.pow(-1).diag_embed()
        sigma_sq = s.pow(2)
        F = sigma_sq.unsqueeze(-2) - sigma_sq.unsqueeze(-1)
        F.diagonal(dim1=-2, dim2=-1).fill_(torch.inf)
        F = F.pow(-1)
        
        u_term = torch.zeros_like(A, device=A.device, dtype=A.dtype)
        v_term = torch.zeros_like(A, device=A.device, dtype=A.dtype)
        
        if u.requires_grad:
            u_term = torch.matmul(u, torch.matmul(F * (torch.matmul(ut, u) - torch.matmul(u.transpose(-2, -1), u)), sigma_mat))
            if m > k:
                tmp = torch.matmul(u, sigma_mat_inv)
                u_term = u_term + (tmp - torch.matmul(u, torch.matmul(ut, tmp)))
            u_term = torch.matmul(u_term, v.transpose(-2, -1))
        
        if v.requires_grad:
            gvt = v.transpose(-2, -1)
            v_term = torch.matmul(sigma_mat, torch.matmul(F * (torch.matmul(v, gvt) - torch.matmul(gvt, v)), v.transpose(-2, -1)))
            if n > k:
                tmp = torch.matmul(gvt, in_n - torch.matmul(v, v.transpose(-2, -1)))
                v_term = v_term + torch.matmul(sigma_mat_inv, tmp)
            v_term = torch.matmul(u, v_term)
        
        return u_term + v_term

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    m, n = 100, 50  # Example dimensions for a tall matrix (m >> n)
    return torch.rand(m, n, dtype=torch.float32)

