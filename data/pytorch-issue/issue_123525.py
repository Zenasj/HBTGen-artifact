# torch.rand(B, 10, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.utils as nn_utils

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # InstanceNorm1d with valid and invalid eps values
        self.norm_correct = nn.InstanceNorm1d(5, eps=1e-5, affine=True)
        self.norm_faulty = nn.InstanceNorm1d(5, eps=2018915490, affine=True)
        
        # SpectralNorm Linear layers with valid and invalid eps values
        self.linear_correct = nn_utils.spectral_norm(
            nn.Linear(10, 5), eps=1e-5
        )
        self.linear_faulty = nn_utils.spectral_norm(
            nn.Linear(10, 5), eps=2018915490
        )
    
    def forward(self, x):
        # Process InstanceNorm1d (requires input reshape to (batch, channels, length))
        x_reshaped = x.view(x.size(0), 5, 2)  # Split 10 features into 5 channels × 2 length
        out_norm_correct = self.norm_correct(x_reshaped)
        out_norm_faulty = self.norm_faulty(x_reshaped)
        norm_diff = torch.abs(out_norm_correct - out_norm_faulty).max() > 1e-6
        
        # Process SpectralNorm Linear layer (direct input)
        out_linear_correct = self.linear_correct(x)
        out_linear_faulty = self.linear_faulty(x)
        linear_nan = torch.isnan(out_linear_faulty).any()
        
        # Return True if any discrepancy exists between valid/invalid eps cases
        return norm_diff or linear_nan

def my_model_function():
    return MyModel()

def GetInput():
    # Input shape (3,10) matches the spectral norm example's input shape
    return torch.randn(3, 10)

