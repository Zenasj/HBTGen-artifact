# torch.rand(3, dtype=torch.float64)
import torch
import numpy as np
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(3, 1, dtype=torch.double)  # Matches input dtype (float64)

    def forward(self, x):
        return self.fc(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a tensor compatible with MyModel (shape (3,), dtype float64)
    arr = np.array([1, 2, 3], dtype=np.float64)
    return torch.from_numpy(arr)  # Automatically creates a DoubleTensor (float64)

