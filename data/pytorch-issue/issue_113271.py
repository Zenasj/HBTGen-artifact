# torch.rand(2, 3, 4, 5, dtype=torch.float32)
import torch
import torch.nn as nn

class ModelA(nn.Module):
    def forward(self, x):
        return x.add_(1)

class ModelB(nn.Module):
    def forward(self, x):
        return x + 1

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.modelA = ModelA()  # in-place modification
        self.modelB = ModelB()  # out-of-place computation

    def forward(self, x):
        inputA = x.clone()  # Ensure independent inputs for both models
        inputB = x.clone()
        outA = self.modelA(inputA)
        outB = self.modelB(inputB)
        return (outA - outB).abs().sum()  # Return error metric between outputs

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 3, 4, 5, dtype=torch.float32)

