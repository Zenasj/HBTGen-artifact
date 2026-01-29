# torch.rand(1, 1, 1, 1, dtype=torch.float32).cuda()
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.param = nn.Parameter(torch.tensor([1.0]).cuda().half())  # Half-precision parameter on CUDA

    def forward(self, x):
        # Forward pass does not use the parameter to isolate the gradient assignment issue
        return x

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a minimal 4D tensor matching the comment's shape requirement
    return torch.randn(1, 1, 1, 1).cuda()

