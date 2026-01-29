# torch.rand(1, 3, 48, 48, dtype=torch.float32) ← Add a comment line at the top with the inferred input shape

import torch
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub
import torch.quantization

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.relu1 = nn.ReLU()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.dequant(x)
        return x

    def fuse_model(self):
        torch.quantization.fuse_modules(self, ['conv1', 'bn1', 'relu1'], inplace=True)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    model = MyModel()
    model.train()
    model.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
    torch.quantization.prepare_qat(model, inplace=True)
    return model

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn((1, 3, 48, 48), dtype=torch.float32)

