# torch.rand(2, 4, 4, dtype=torch.float)  # Batch of 2 4x4 matrices
import torch
import numpy as np
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Precompute correct inverse using numpy as a reference
        mat1 = np.diag([1, 0.5, 0.25, 0.125])
        mat2 = np.diag([1, 2, 3, 4])
        mats_np = np.array([mat1, mat2])
        correct_inv_np = np.linalg.inv(mats_np)
        # Register as buffer to ensure correct_inv is on the same device as the model
        self.register_buffer('correct_inv', torch.from_numpy(correct_inv_np).float())

    def forward(self, x):
        # Compute inverse using torch.linalg.inv
        inv_torch = torch.linalg.inv(x)
        # Compute error against precomputed correct inverse
        error = inv_torch - self.correct_inv.to(inv_torch.device)
        return error

def my_model_function():
    return MyModel()

def GetInput():
    # Create input tensor matching the test case from the issue
    mat1 = np.diag([1, 0.5, 0.25, 0.125])
    mat2 = np.diag([1, 2, 3, 4])
    input_data = torch.tensor(np.array([mat1, mat2]), dtype=torch.float)
    return input_data

