# torch.rand(1, 3, 224, 224, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.conv1 = nn.Conv2d(3, 6, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(6)
        self.relu = nn.ReLU()
        self.dequant = torch.ao.quantization.DeQuantStub()
        
    def forward(self, x):
        # Quantize input → process → dequantize output
        x = self.quant(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dequant(x)
        return x

def my_model_function():
    model = MyModel()
    # Initialize weights (default PyTorch initialization)
    return model

def GetInput():
    # Random input matching expected shape (B=1, C=3, H=224, W=224)
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

