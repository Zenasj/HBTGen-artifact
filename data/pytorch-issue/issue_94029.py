# torch.randint(10, 30, (10,), dtype=torch.long)  # three inputs
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, inputs):
        a, b, c = inputs
        return a * b * c

def my_model_function():
    return MyModel()

def GetInput():
    a = torch.randint(10, 30, (10,), dtype=torch.long)
    return (a, a, a)

