# torch.rand(B, 3, 224, 224, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(3 * 224 * 224, 10)  # Dummy layer for demonstration

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

