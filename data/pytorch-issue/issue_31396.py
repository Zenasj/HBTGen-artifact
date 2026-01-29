# torch.rand(14650, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(14650, 2)
        self.output_activ = nn.Softmax(dim=-1)
    
    def forward(self, x):
        x = self.input(x)
        return self.output_activ(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(14650, dtype=torch.float32)

