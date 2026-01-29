# torch.rand(B, dtype=torch.float)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(1, 1, bias=False)
    
    def forward(self, x):
        return self.linear(x)

def my_model_function():
    model = MyModel()
    # Initialize weights to match example behavior (optional but consistent)
    with torch.no_grad():
        model.linear.weight.fill_(0.5)
    return model

def GetInput():
    # Returns a single-element tensor matching the model's input expectation
    return torch.rand(1, dtype=torch.float)

