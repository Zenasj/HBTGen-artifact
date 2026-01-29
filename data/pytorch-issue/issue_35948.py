# torch.rand(B, C, H, W, dtype=torch.float32)  # Example input shape: (1, 3, 224, 224)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.string_param = "5"  # String parameter to trigger int() conversion in forward

    def forward(self, x):
        val = int(self.string_param)  # This line will fail when scripted due to str->int conversion
        return x + val  # Example operation using the converted integer

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

