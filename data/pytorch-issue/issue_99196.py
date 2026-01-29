# torch.rand(B, C, H, W, dtype=...)  # (32, 8, 256)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(in_features=256, out_features=128, bias=False)

    def forward(self, x):
        return self.linear(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn(32, 8, 256).cuda()

