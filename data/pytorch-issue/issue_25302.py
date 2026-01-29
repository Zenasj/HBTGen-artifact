# torch.rand(B, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # Simple model to process 1D input
        
    def forward(self, x):
        # Input x is (batch_size,)
        x = x.unsqueeze(1)  # Reshape to (batch_size, 1)
        return self.linear(x)

def my_model_function():
    # Returns a simple model instance with default initialization
    return MyModel()

def GetInput():
    # Returns a random tensor matching the model's expected input shape
    return torch.rand(16, dtype=torch.float32)  # Batch size 16 (from original example)

