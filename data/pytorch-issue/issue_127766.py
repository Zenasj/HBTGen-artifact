# torch.rand(10, requires_grad=True) ← Add a comment line at the top with the inferred input shape
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.mod1 = nn.Linear(10, 10)
        self.mod2 = nn.Linear(10, 10)

    def forward(self, x):
        return self.mod2(self.mod1(x))

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(10, requires_grad=True)

