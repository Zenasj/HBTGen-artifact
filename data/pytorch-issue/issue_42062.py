# torch.rand(3000, 1000, dtype=torch.float)  # Inferred input shape based on Reddit dataset structure (n=1000, m=10000)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        # Apply torch.svd_lowrank to sparse input, which may trigger memory issues
        U, S, V = torch.svd_lowrank(x)
        return S  # Return singular values as the model output

def my_model_function():
    return MyModel()

def GetInput():
    # Generate sparse input tensor mimicking Reddit dataset structure
    n = 1000  # Example number of nodes
    m = 10000  # Example number of edges
    indices = torch.cat([
        torch.randint(0, n, (2, m)),          # Original edges
        n + torch.randint(0, n, (2, m)),      # Second copy shifted by n
        2 * n + torch.randint(0, n, (2, m))   # Third copy shifted by 2n
    ], dim=-1)
    values = torch.ones(3 * m).float()
    # Create sparse tensor with shape (3n, n) to match Reddit example's structure
    sparse_tensor = torch.sparse_coo_tensor(
        indices, values, (3 * n, n)
    ).coalesce()  # Ensure coalesced format for compatibility
    return sparse_tensor

