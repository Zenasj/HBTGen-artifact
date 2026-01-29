# torch.rand(B, 10, 32, dtype=torch.float32)  # Input shape for each tensor in the tuple

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Linear(32, 16)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        x1, x2 = x
        x1 = self.conv1(x1)
        x1 = self.relu1(x1)
        x2 = self.conv1(x2)
        x2 = self.relu1(x2)
        out = torch.cat((x1, x2), dim=-1)
        out = self.fc(out)
        return out

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    B = 2
    x1 = torch.randn((B, 10, 32))
    x2 = torch.randn((B, 10, 32))
    return (x1, x2)

