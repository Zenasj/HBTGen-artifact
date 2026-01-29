# torch.rand(B, 4, dtype=torch.float), torch.rand(B, 4, dtype=torch.float)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, inputs):
        x, h = inputs  # Unpack tuple from GetInput()
        new_h = torch.tanh(self.linear(x) + h)
        return new_h, new_h  # Matches original output signature

def my_model_function():
    return MyModel()

def GetInput():
    B = 1  # Arbitrary batch size (can be adjusted)
    x = torch.rand(B, 4, dtype=torch.float)
    h = torch.rand(B, 4, dtype=torch.float)
    return (x, h)  # Returns tuple matching forward() input requirements

