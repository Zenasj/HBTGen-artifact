# torch.rand(1, 2, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_cloned = torch.clone(x)  # Replaced copy.copy with torch.clone per fix suggestion
        return self.relu(x_cloned)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 2)

