# torch.rand(1, 3, 32, 32, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.conv = nn.Conv2d(3, 1, kernel_size=3)
        self.relu = nn.ReLU()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.relu(self.conv(x))
        x = torch.cat([x, x], dim=1)  # Concatenation along channel axis
        x = self.dequant(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1, 3, 32, 32)

