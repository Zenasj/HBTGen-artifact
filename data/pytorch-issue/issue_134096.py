import torch
import torch.nn as nn

# torch.rand(B, C, H, W, dtype=torch.float32)
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        return self.avg_pool(x)

def my_model_function():
    # Returns a model instance using the overridden avg_pool2d operator
    return MyModel()

def GetInput():
    # Returns a random tensor compatible with RISC-V's avg_pool2d implementation
    return torch.rand(1, 3, 32, 32, dtype=torch.float32)

