# torch.rand(1, 8, 546, 392, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(8, 1, kernel_size=3, padding=0)  # Matches the problematic layer

    def forward(self, x):
        return self.conv(x)

def my_model_function():
    model = MyModel()
    model.eval()  # Matches the user's eval() setup
    model.cuda()  # Matches CUDA execution context
    return model

def GetInput():
    # Returns a float32 tensor on CUDA, compatible with autocast's fp16 conversion
    return torch.rand(1, 8, 546, 392, dtype=torch.float32, device="cuda")

