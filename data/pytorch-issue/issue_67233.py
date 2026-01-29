# torch.rand(B=1, C=3, H=224, W=224, dtype=torch.float32)  # Common image input shape assumption
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Dummy model to demonstrate autocast behavior
        self.linear = nn.Linear(224*224*3, 1)  # Arbitrary operations to trigger autocast
        
    def forward(self, x):
        # Flatten for linear layer (for demonstration purposes)
        x = x.view(x.size(0), -1)
        return self.linear(x)

def my_model_function():
    # Initialize model with dummy weights
    model = MyModel()
    return model

def GetInput():
    # Generate random input matching assumed shape
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

