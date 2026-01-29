# torch.rand(B, 1, 1, 1, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.L = nn.Linear(1, 50)
        self.out = nn.Linear(50, 1)

    def forward(self, x):
        # Flatten 4D input (B, C, H, W) to 2D (B, features) for linear layers
        x = x.view(x.size(0), -1)
        Y = torch.tanh(self.L(x))
        Y = self.out(Y)
        return Y

def my_model_function():
    # Initialize the model with default parameters
    return MyModel()

def GetInput():
    # Generate random input tensor matching expected shape (B, 1, 1, 1)
    B = 10  # Example batch size
    return torch.rand(B, 1, 1, 1, dtype=torch.float32)

