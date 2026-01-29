# torch.rand(B, 512, 512, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        # Compare logdet and slogdet outputs to detect instability
        sign, slogdet_logdet = torch.slogdet(x)
        logdet = torch.logdet(x)
        # Return True if outputs match within tolerance (handles numerical issues)
        return torch.isclose(logdet, slogdet_logdet, atol=1e-8, rtol=1e-5).all()

def my_model_function():
    return MyModel()

def GetInput():
    # Create scaled identity matrix input (replicates issue conditions)
    B, N = 1, 512  # Batch size and matrix dimensions from issue example
    diag = torch.rand(N) / 500000  # Small diagonal values to trigger instability
    x = torch.diag_embed(diag).unsqueeze(0)  # Add batch dimension
    return x

