# torch.rand(10, 6, 224, 224, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.batch_norm = nn.BatchNorm2d(num_features=6)  # Matches the minimal repro example
    
    def forward(self, x):
        return self.batch_norm(x)

def my_model_function():
    return MyModel()  # Returns initialized BatchNorm2d model

def GetInput():
    return torch.randn(10, 6, 224, 224)  # Matches input dimensions from issue's minimal example

