# torch.rand(B, 10, dtype=torch.float32)  # Input shape inferred as (batch, features)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # This torch.device(0) call triggers CUDA initialization in the main process
        # leading to potential CUDA fork issues in subprocesses (as described in the issue)
        self.device = torch.device(0)  
        self.fc = nn.Linear(10, 5)  # Example layer

    def forward(self, x):
        # Forward pass moves input to CUDA device (triggering subprocess errors if initialized improperly)
        return self.fc(x.to(self.device))

def my_model_function():
    # Returns a model instance with problematic CUDA initialization in __init__
    return MyModel()

def GetInput():
    # Generates a random input tensor matching the expected shape (B=1, features=10)
    return torch.rand(1, 10, dtype=torch.float32)

