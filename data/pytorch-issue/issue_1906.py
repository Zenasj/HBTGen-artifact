# torch.rand(1, 3, 448, 448, dtype=torch.float32)  # Inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.max_pool = nn.MaxPool2d(2)
        self.avg_pool = nn.AvgPool2d(2)

    def forward(self, x):
        max_pooled = self.max_pool(x)
        avg_pooled = self.avg_pool(x)
        return max_pooled, avg_pooled

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(1, 3, 448, 448, dtype=torch.float32)

