# torch.rand(B=1, C=1, H=1, W=1, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Simple model to demonstrate distributed setup compatibility
        self.linear = nn.Linear(1, 1)
        
    def forward(self, x):
        # Flatten input for linear layer compatibility
        return self.linear(x.view(x.size(0), -1))

def my_model_function():
    # Initialize model with random weights
    model = MyModel()
    return model

def GetInput():
    # Generate input matching the model's expected dimensions
    return torch.rand(1, 1, 1, 1, dtype=torch.float32).cuda()

