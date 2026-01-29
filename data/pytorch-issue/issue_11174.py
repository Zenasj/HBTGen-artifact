# torch.rand(M, N_plus_1, dtype=torch.float32)  # Input shape: (M, N+1), where N is the number of columns in matrix A

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        M, N_plus_1 = x.shape
        N = N_plus_1 - 1
        A = x[:, :N]
        b = x[:, N]
        
        # Compute solutions using different LAPACK drivers
        try:
            sol_gels = torch.linalg.lstsq(A, b, driver='gels').solution
        except RuntimeError:
            sol_gels = torch.full((N,), float('nan'), device=x.device)
        
        try:
            sol_gelsd = torch.linalg.lstsq(A, b, driver='gelsd').solution
        except RuntimeError:
            sol_gelsd = torch.full((N,), float('nan'), device=x.device)
        
        # Check for numerical instability in GELS solution
        has_nans = torch.isnan(sol_gels).any() or torch.isinf(sol_gels).any()
        # Compare solutions within tolerance
        close = torch.allclose(sol_gels, sol_gelsd, atol=1e-5)
        return torch.tensor(not has_nans and close, dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    # Create rank-deficient matrix A (M rows, N cols) and random vector b (M elements)
    M = 1000
    N = 600
    rank = N // 2  # Ensure rank-deficient matrix
    
    U = torch.rand(M, rank)
    V = torch.rand(N, rank)
    A = U @ V.T  # Rank-deficient matrix
    
    b = torch.rand(M)
    
    # Combine into single input tensor
    input_tensor = torch.cat([A, b.unsqueeze(1)], dim=1)
    return input_tensor

