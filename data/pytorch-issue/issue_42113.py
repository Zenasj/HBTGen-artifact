# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub

class DownBlockQ(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.quant_input = QuantStub()
        self.dequant_output = DeQuantStub()

        self.conv1 = nn.Conv2d(in_ch, in_ch, 4, stride=2, padding=1, groups=in_ch)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_ch, out_ch, 1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        # x = self.quant_input(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        # x = self.dequant_output(x)
        return x

    def fuse_model(self):
        torch.quantization.fuse_modules(self, ['conv1', 'bn1', 'relu1'], inplace=True)
        torch.quantization.fuse_modules(self, ['conv2', 'bn2', 'relu2'], inplace=True)

class MyModel(nn.Module):
    def __init__(self, filters=22):
        super().__init__()
        self.quant_input = QuantStub()
        self.dequant_output = DeQuantStub()

        self.db1 = DownBlockQ(filters * 1, filters * 2)
        self.db2 = DownBlockQ(filters * 2, filters * 4)
        self.db3 = DownBlockQ(filters * 4, filters * 8)

    def forward(self, x):
        x = self.quant_input(x)
        x = self.db1(x)
        x = self.db2(x)
        x = self.db3(x)
        x = self.dequant_output(x)
        return x

def my_model_function():
    return MyModel(filters=22)

def GetInput():
    return torch.rand(1, 22, 256, 256, dtype=torch.float32)

def fuse_model(model):
    if hasattr(model, 'fuse_model'):
        model.fuse_model()
    for p in list(model.modules())[1:]:
        fuse_model(p)

