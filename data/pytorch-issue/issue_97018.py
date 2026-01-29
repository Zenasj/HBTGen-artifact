# torch.rand(1, 3, 224, 224, dtype=torch.half)
import torch
from torchvision.models import convnext_base
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = convnext_base()  # Standard ConvNeXt base architecture

    def forward(self, x):
        return self.model(x)

def my_model_function():
    # Initialize model with CUDA and half-precision to match the issue's setup
    model = MyModel().cuda().half()
    return model

def GetInput():
    # Generate input tensor matching ConvNeXt's expected input shape and dtype
    return torch.randn((1, 3, 224, 224), device="cuda", dtype=torch.half)

