# torch.rand(B, 100)  # Assuming input shape is batch x 100 features
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(100, 256)
        self.bn = nn.BatchNorm1d(256)  # Layer causing device mismatch error
        self.fc2 = nn.Linear(256, 3 * 32 * 32)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn(x)  # BatchNorm weight parameter referenced in error
        x = self.fc2(x)
        return x.view(x.size(0), 3, 32, 32)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(16, 100)  # Batch size 16, 100 features (matches input comment)

