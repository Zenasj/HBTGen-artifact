# torch.rand(B, 10, dtype=torch.float32)  # Input shape inferred from Linear(10, 5)
import torch
import numpy as np
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.net = nn.Linear(10, 5)  # Matches input shape (batch, 10)
        self.class_p = nn.Parameter(torch.Tensor(np.ones(81) * np.log(1.0)), requires_grad=True)
        self.class_p_t = self.class_p.data  # Problematic assignment causing multiprocessing issues

    def forward(self, x):
        return self.net(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(20, 10, dtype=torch.float32)  # Matches example input in error reproduction script

