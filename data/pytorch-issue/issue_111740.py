# torch.rand(4, 2048, 512, 1, dtype=torch.float32)  # Input shape (B, S, F, 1). Actual input is 3D (B, S, F) with W=1 for structural compliance
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.v_proj = nn.Linear(512, 128, bias=False)  # Matches the original Linear layer configuration
    
    def forward(self, x):
        x = x.squeeze(-1)  # Convert 4D (B, S, F, 1) to 3D (B, S, F) for compatibility
        full = self.v_proj(x)
        chunks = torch.chunk(x, 2, dim=0)  # Split into two chunks along batch dimension
        split = torch.cat([self.v_proj(c) for c in chunks], dim=0)  # Process chunks individually and concatenate
        # Return maximum absolute difference between full and split outputs
        return torch.max(torch.abs(full - split))

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(4, 2048, 512, 1, dtype=torch.float32)  # Matches the structural input format requirement

