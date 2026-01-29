# torch.rand(1, 3, 224, 224, dtype=torch.float32)  # Inferred input shape

import torch
import torch.nn as nn

class QuantizedConvReLU2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, scale, zero_point, padding=0):
        super(QuantizedConvReLU2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = scale
        self.zero_point = zero_point
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = (x / self.scale) + self.zero_point
        x = self.relu(x)
        return x

class QuantizedConvAddReLU2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, scale, zero_point, padding=0):
        super(QuantizedConvAddReLU2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = scale
        self.zero_point = zero_point
        self.relu = nn.ReLU()

    def forward(self, x, residual):
        x = self.conv(x)
        x = (x / self.scale) + self.zero_point
        x = x + residual
        x = self.relu(x)
        return x

class QuantizedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, scale, zero_point, padding=0):
        super(QuantizedConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = scale
        self.zero_point = zero_point

    def forward(self, x):
        x = self.conv(x)
        x = (x / self.scale) + self.zero_point
        return x

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = QuantizedConvReLU2d(3, 64, kernel_size=(3, 3), stride=(1, 1), scale=0.013380219228565693, zero_point=128, padding=(1, 1))
        
        self.layer1_0_conv1 = QuantizedConvReLU2d(64, 64, kernel_size=(1, 1), stride=(1, 1), scale=0.005257370416074991, zero_point=128)
        self.layer1_0_conv2 = QuantizedConvReLU2d(64, 64, kernel_size=(3, 3), stride=(1, 1), scale=0.008329497650265694, zero_point=128, padding=(1, 1))
        self.layer1_0_conv3 = QuantizedConvAddReLU2d(64, 256, kernel_size=(1, 1), stride=(1, 1), scale=0.010099029168486595, zero_point=128)
        self.layer1_0_downsample = QuantizedConv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), scale=0.005250006914138794, zero_point=128)

        self.layer1_1_conv1 = QuantizedConvReLU2d(256, 64, kernel_size=(1, 1), stride=(1, 1), scale=0.005096939858049154, zero_point=128)
        self.layer1_1_conv2 = QuantizedConvReLU2d(64, 64, kernel_size=(3, 3), stride=(1, 1), scale=0.004832049366086721, zero_point=128, padding=(1, 1))
        self.layer1_1_conv3 = QuantizedConvAddReLU2d(64, 256, kernel_size=(1, 1), stride=(1, 1), scale=0.00993015430867672, zero_point=128)

        self.layer1_2_conv1 = QuantizedConvReLU2d(256, 64, kernel_size=(1, 1), stride=(1, 1), scale=0.0038794134743511677, zero_point=128)
        self.layer1_2_conv2 = QuantizedConvReLU2d(64, 64, kernel_size=(3, 3), stride=(1, 1), scale=0.005039558280259371, zero_point=128, padding=(1, 1))
        self.layer1_2_conv3 = QuantizedConvAddReLU2d(64, 256, kernel_size=(1, 1), stride=(1, 1), scale=0.010828116908669472, zero_point=128)

        # Add more layers as per the model structure

    def forward(self, x):
        x = self.conv1(x)
        
        residual = x
        x = self.layer1_0_conv1(x)
        x = self.layer1_0_conv2(x)
        x = self.layer1_0_conv3(x, self.layer1_0_downsample(residual))

        residual = x
        x = self.layer1_1_conv1(x)
        x = self.layer1_1_conv2(x)
        x = self.layer1_1_conv3(x, residual)

        residual = x
        x = self.layer1_2_conv1(x)
        x = self.layer1_2_conv2(x)
        x = self.layer1_2_conv3(x, residual)

        # Add more forward passes for the remaining layers

        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1, 3, 224, 224, dtype=torch.float32)

