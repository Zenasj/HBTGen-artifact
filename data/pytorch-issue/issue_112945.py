# torch.rand(B, 150528, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(3 * 224 * 224, 64)  # Input shape 3x224x224 flattened
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 64)
        self.bn2 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3 * 224 * 224)  # Batch size 1, input shape (150528,)

