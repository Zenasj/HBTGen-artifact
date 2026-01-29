# torch.rand(2, dtype=torch.float32)
import torch
from torch import nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.B = nn.Parameter(torch.zeros(2, requires_grad=True))  # Target with requires_grad
        self.C = nn.Parameter(torch.zeros(2, requires_grad=False))  # Target without requires_grad
    
    def forward(self, A):
        # Compute MSE loss for both targets and compare reduction behavior
        loss_b = F.mse_loss(A, self.B, reduction='elementwise_mean')
        loss_c = F.mse_loss(A, self.C, reduction='elementwise_mean')
        # Return 1.0 if losses differ (indicating the bug's discrepancy), else 0.0
        return (loss_b != loss_c).float()

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, dtype=torch.float32)

