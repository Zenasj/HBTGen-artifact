# torch.rand(32, 64, 1024, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize indices within valid ranges (x: 0-63, y: 0-1023)
        self.x_indices = nn.Parameter(torch.randint(0, 64, (50000,), dtype=torch.long), requires_grad=False)
        self.y_indices = nn.Parameter(torch.randint(0, 1024, (50000,), dtype=torch.long), requires_grad=False)
    
    def forward(self, data):
        B, H, W = data.shape
        indices_linear = self.x_indices * W + self.y_indices  # Compute linear indices
        indices_linear = indices_linear.unsqueeze(0).expand(B, -1)  # Expand to batch dimension
        data_flat = data.view(B, H * W)  # Flatten last two dimensions
        return data_flat.gather(1, indices_linear)  # Gather along flattened dimension

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(32, 64, 1024, dtype=torch.float32)

