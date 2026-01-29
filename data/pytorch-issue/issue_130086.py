# torch.rand(0, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        # Compare two approaches: max with dim=0 vs max with another tensor
        # Return 1.0 if error occurs in first approach, else 0.0
        error_flag = torch.tensor(0.0)
        try:
            # Case 1: Single tensor with dim=0 (may error)
            _ = torch.max(x, dim=0)
        except:
            error_flag = torch.tensor(1.0)
        
        # Case 2: Two tensors (no dim needed, works for empty 1D)
        _ = torch.max(x, x)  # Always works for empty 1D
        
        return error_flag  # 1.0 indicates error in Case 1

def my_model_function():
    return MyModel()

def GetInput():
    # Input that triggers error in Case 1 but works in Case 2
    return torch.tensor([], dtype=torch.float32)

