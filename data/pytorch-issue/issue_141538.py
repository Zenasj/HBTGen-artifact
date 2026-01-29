# torch.rand(1, 1, 10, 10, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.FractionalMaxPool2d(kernel_size=(1, 1), output_ratio=(0.5, 0.5))
    
    def forward(self, x):
        return self.pool(x)

def my_model_function():
    # Return an instance in eval mode as in the issue's setup
    return MyModel().eval()

def GetInput():
    # Matches the input shape and dtype used in the original test case
    return torch.randn(1, 1, 10, 10)

