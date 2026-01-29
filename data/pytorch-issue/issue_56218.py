# torch.rand(1, 3, 224, 224, dtype=torch.float32)  # Inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.avg_pool2d = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2), count_include_pad=False)

    def forward(self, x):
        x = self.avg_pool2d(x)
        return x

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

