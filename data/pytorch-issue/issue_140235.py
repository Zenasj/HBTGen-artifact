# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=3)
        # Test case from issue: set dilation as integer (now handled by the PR fix)
        self.conv.dilation = 1

    def forward(self, x):
        return self.conv(x)

def my_model_function():
    # Returns the fixed model using the PR's Conv2d implementation
    return MyModel()

def GetInput():
    # Matches input shape from issue's test case (1,1,5,5)
    return torch.randn(1, 1, 5, 5, dtype=torch.float32)

