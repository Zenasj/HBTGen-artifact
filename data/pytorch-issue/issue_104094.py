# torch.rand(B, 1, dtype=torch.float)  # Input shape (B, 1) for MyModel
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Simple linear layer to produce output matching the example's 2-element prediction
        self.fc = nn.Linear(1, 2)  # Output dim=2 as per user's example [1,2]
    
    def forward(self, x):
        return self.fc(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Generate random input with shape (B, 1) where B is batch size
    B = 1  # Example batch size
    return torch.rand(B, 1, dtype=torch.float)

