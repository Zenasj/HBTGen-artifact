# torch.rand(B, 2, 3, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(2 * 3, 10)  # Input shape (2,3) → flattened to 6 features

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input tensor
        return self.fc(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a random tensor matching the input expected by MyModel
    return torch.rand(4, 2, 3)  # B=4, channels=2, spatial dimensions=3 (e.g., 1x3)

