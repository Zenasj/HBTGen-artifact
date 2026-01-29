# torch.rand(1, 1, 1024, 1024, dtype=torch.float32).cuda()
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        # Reproduces SVD convergence issue leading to matrix_rank exception
        return torch.linalg.matrix_rank(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Create singular matrix with last row and column zeroed
    a = torch.rand(1, 1, 1024, 1024, dtype=torch.float32).cuda()
    a[..., -1, :] = 0  # Zero last row
    a[..., :, -1] = 0  # Zero last column
    return a

