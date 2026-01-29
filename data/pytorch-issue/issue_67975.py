# torch.rand(B, N, N, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # LowerTriangular comparison (original vs proposed)
        value_tril = x.tril()
        mask_orig = (value_tril == x)
        flat_orig = mask_orig.view(x.shape[:-2] + (-1,))
        lt_orig = flat_orig.min(-1).values  # Original approach

        lt_prop = mask_orig.all(dim=(-2, -1))  # Proposed approach
        lt_agree = (lt_orig == lt_prop).all()

        # LowerCholesky comparison (original vs proposed)
        diag = x.diagonal(dim1=-2, dim2=-1)
        lower_triangular_orig = (value_tril == x).view(x.shape[:-2] + (-1,)).min(-1).values
        positive_diag_orig = (diag > 0).min(-1).values
        lc_orig = lower_triangular_orig & positive_diag_orig

        lower_triangular_prop = (value_tril == x).all(dim=(-2, -1))
        positive_diag_prop = (diag > 0).all(dim=-1)
        lc_prop = lower_triangular_prop & positive_diag_prop
        lc_agree = (lc_orig == lc_prop).all()

        return torch.tensor([lt_agree, lc_agree], dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    B, N = 2, 3  # Batch size and matrix size
    return torch.rand(B, N, N, dtype=torch.float32)

