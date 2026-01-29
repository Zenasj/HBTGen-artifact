# torch.rand(B, 3, 224, 224, dtype=torch.float32)
import torch
import torch.nn as nn
from torchvision.models import convnext_small
from torch.ao.quantization import QuantStub, DeQuantStub

class MyModel(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.model = convnext_small(weights="DEFAULT")
        # Replace last layer to match custom num_classes
        self.model.classifier[2] = nn.Linear(
            in_features=self.model.classifier[2].in_features,
            out_features=num_classes
        )
        # Quantization stubs required for QAT
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        # Quantize input
        x = self.quant(x)
        # Forward through main model
        x = self.model(x)
        # Dequantize output
        x = self.dequant(x)
        return x

def my_model_function():
    # Initialize with default ImageNet classes (1000)
    return MyModel()

def GetInput():
    # Generate random input matching ConvNeXt Small's expected input
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

