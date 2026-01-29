# torch.rand(B, 3, 3, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
    def forward(self, x):
        # Compute trace of input matrices
        diag_x = torch.diagonal(x, 0, 1, 2)  # Get diagonal along last two dims
        sum_diag_x = torch.sum(diag_x, dim=1)  # Sum to get trace
        
        # Compute matrix product followed by trace
        matmul_xx = torch.matmul(x, x)  # Batch matrix multiply
        diag_matmul = torch.diagonal(matmul_xx, 0, 1, 2)
        sum_diag_matmul = torch.sum(diag_matmul, dim=1)
        
        return sum_diag_x, sum_diag_matmul  # Return both sums as outputs

def my_model_function():
    return MyModel()

def GetInput():
    B = 100  # Batch size from issue example
    return torch.rand(B, 3, 3, dtype=torch.float32)

