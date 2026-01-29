# torch.rand(1, 3, 224, 224, dtype=torch.half)  # Inferred input shape from half-precision use case
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
        self.fc = nn.Linear(16 * 222 * 222, 10)  # 224-2=222 after conv without padding

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def my_model_function():
    model = MyModel()
    model.half()  # Enforce half-precision operations requiring CUDA flags
    return model

def GetInput():
    return torch.rand(1, 3, 224, 224, dtype=torch.half)  # Matches model's expected input

