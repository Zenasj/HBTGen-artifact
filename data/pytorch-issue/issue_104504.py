# torch.rand(B, 32, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(32, 64)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        return self.layer2(x)

def my_model_function():
    # Returns a simple FFNN model with parameters suitable for FSDP testing
    return MyModel()

def GetInput():
    # Generate a random input tensor matching the model's expected input shape
    B = 2  # Batch size
    return torch.rand(B, 32, dtype=torch.float32)

