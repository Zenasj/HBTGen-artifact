import torch.nn as nn

import copy
import torch
import torch._dynamo

class FusedQATConvBnPattern(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1)
        self.bn = torch.nn.BatchNorm2d(1)

    def forward(self, input):
        """Approximated method to fuse conv and bn. It requires only one forward pass.
        conv_orig = conv / scale_factor where scale_factor = bn.weight / running_std
        """
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

example_inputs = (torch.randn(1, 1, 3, 3),)

pattern, _ = torch._dynamo.export(
    FusedQATConvBnPattern(),
    *copy.deepcopy(example_inputs),
    aten_graph=True,
    tracing_mode="real",
)