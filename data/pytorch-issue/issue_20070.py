# torch.rand(5, 3, dtype=torch.float)  # inferred input shape (B=5, features=3)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(3, 2)  # Example layer matching input features
        
    def forward(self, x):
        return self.linear(x)

def my_model_function():
    model = MyModel()
    # Initialize weights (optional, but ensures determinism)
    with torch.no_grad():
        model.linear.weight.fill_(1.0)
        model.linear.bias.fill_(0.0)
    return model

def GetInput():
    return torch.rand(5, 3, dtype=torch.float)  # Valid input with float dtype

