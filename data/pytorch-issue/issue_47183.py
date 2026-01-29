# torch.rand(4, 3, 20, 20, dtype=torch.float32)  # Add a comment line at the top with the inferred input shape
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)
        self.in_conv = nn.Conv2d(3, 10, 3, 1, 1)
        self.out_conv = nn.Conv2d(10, 1, 1)

    def forward(self, x):
        x = checkpoint(self.ck_in_conv, x, self.dummy_tensor)
        x = checkpoint(self.ck_out_conv, x)
        return x

    def ck_in_conv(self, x, dummy_arg=None):
        assert dummy_arg is not None
        x = self.in_conv(x)
        return x

    def ck_out_conv(self, x):
        x = self.out_conv(x)
        return x

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(4, 3, 20, 20, dtype=torch.float32).cuda()

