# torch.rand(B, 128, 768, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(768, 100000)

    def forward(self, x):
        x = x * 2
        x = x * 2
        x = x * 2
        x = self.fc(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(10, 128, 768, requires_grad=True, dtype=torch.float32)

