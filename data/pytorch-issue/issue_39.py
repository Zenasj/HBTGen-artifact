# torch.rand(B, C, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(100, 50)  # Matches input dimension from GetInput()
        self.bn = nn.BatchNorm1d(50)  # Resolved std::bad_cast issue via proper initialization
        self.activation = nn.ReLU()
        
    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        return self.activation(x)

def my_model_function():
    # Returns a model with initialized weights and layers
    return MyModel()

def GetInput():
    # Returns a 2D tensor matching the Linear layer's input expectation
    return torch.rand(32, 100, dtype=torch.float32)  # B=32, C=100 (matches Linear(100, 50))

