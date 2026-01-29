# torch.rand(B, 3, requires_grad=True)  # Input shape inferred from model's first layer (3 features)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.lin1 = nn.Linear(3, 30)
        self.lin2 = nn.Linear(30, 1)

    def forward(self, p):
        x = self.lin1(p)
        x = nn.ReLU()(x)
        return self.lin2(x)

def my_model_function():
    # Returns the model instance with default initialization
    return MyModel()

def GetInput():
    # Returns a random tensor with requires_grad=True (required for gradient computation)
    return torch.rand(100, 3, requires_grad=True)

