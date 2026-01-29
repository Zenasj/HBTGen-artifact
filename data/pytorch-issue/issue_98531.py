# torch.rand(B, 1, 3, 3, dtype=torch.float32)  # Inferred input shape from example_inputs
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, input):
        running_std = torch.sqrt(self.bn.running_var + self.bn.eps)
        scale_factor = self.bn.weight / running_std
        weight_shape = [1] * len(self.conv.weight.shape)
        weight_shape[0] = -1
        bias_shape = [1] * len(self.conv.weight.shape)
        bias_shape[1] = -1
        scaled_weight = self.conv.weight * scale_factor.reshape(weight_shape)
        zero_bias = torch.zeros_like(self.conv.bias, dtype=input.dtype)
        conv = self.conv._conv_forward(input, scaled_weight, zero_bias)
        conv_orig = conv / scale_factor.reshape(bias_shape)
        conv_orig = conv_orig + self.conv.bias.reshape(bias_shape)
        conv = self.bn(conv_orig)
        return conv

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1, 1, 3, 3)

