# torch.rand(B, C, H, W, dtype=torch.float32)  # Assumed input shape (e.g., 1x3x224x224)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.branch1 = nn.Sequential()  # Empty Sequential causing observer retention issue

    def forward(self, x):
        return x  # Minimal forward pass to reproduce quantization conversion bug

def my_model_function():
    model = MyModel()
    model.qconfig = torch.quantization.get_default_qconfig()  # Required for quantization
    return model

def GetInput():
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)  # Typical image input dimensions

