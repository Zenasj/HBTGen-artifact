# torch.rand(B, 200000, 2, dtype=torch.long)  # Input shape inferred from index operations
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Parameter(torch.randn(10000, 64))  # Feature tensor with 10000 entries of 64 dimensions
        
    def forward(self, x):
        # x: (B, 200000, 2) tensor of indices
        indices1 = x[..., 0]  # First set of indices
        indices2 = x[..., 1]  # Second set of indices
        
        # Adjust indices by adding 10000 if negative (mimics C++ code logic)
        adj1 = torch.where(indices1 < 0, indices1 + 10000, indices1)
        adj2 = torch.where(indices2 < 0, indices2 + 10000, indices2)
        
        # Clamp indices to valid range [0, 9999]
        adj1 = torch.clamp(adj1, 0, 9999)
        adj2 = torch.clamp(adj2, 0, 9999)
        
        # Gather features using adjusted indices
        vec1 = self.features[adj1.view(-1)].view(x.size(0), 200000, 64)
        vec2 = self.features[adj2.view(-1)].view(x.size(0), 200000, 64)
        
        # Compute difference (as seen in C++ subtraction between tmp9 and tmp4)
        output = vec2 - vec1
        
        return output

def my_model_function():
    # Returns an instance of MyModel with default initialization
    return MyModel()

def GetInput():
    # Generate random indices within [-10000, 10000) to match adjustment logic
    B = 1  # Batch size inferred from benchmark context
    return torch.randint(-10000, 10000, (B, 200000, 2), dtype=torch.long)

