# torch.rand(B, N, N, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, num_iterations=10):
        super(MyModel, self).__init__()
        self.num_iterations = num_iterations

    def forward(self, A):
        B, N, _ = A.shape
        identity = torch.eye(N, device=A.device, dtype=A.dtype).view(1, N, N).repeat(B, 1, 1)
        
        # Compute L1 norm (max column sum) and Linf norm (max row sum)
        col_sums = A.abs().sum(dim=1)  # (B, N)
        norm1 = col_sums.max(dim=1, keepdim=True).values  # (B, 1)
        
        row_sums = A.abs().sum(dim=2)  # (B, N)
        norm_inf = row_sums.max(dim=1, keepdim=True).values  # (B, 1)
        
        alpha = 1.0 / (norm1 * norm_inf)
        alpha = alpha.view(B, 1, 1)
        N_t = alpha * A.transpose(1, 2)  # Initial N0

        for _ in range(self.num_iterations):
            term1 = torch.bmm(A, N_t)
            term2 = 3 * identity - term1
            term_inner = torch.bmm(term1, term2)
            bracket = 3 * identity - term_inner
            N_t = torch.bmm(N_t, bracket)

        return N_t

def my_model_function():
    return MyModel()

def GetInput():
    # Example input with N=3000 as in original code
    B = 2
    N = 3000
    return torch.rand(B, N, N, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')

