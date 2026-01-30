import torch.nn as nn
import torch.nn.functional as F

self.input_layer = nn.Sequential(
    nn.Conv3d(num_channels, 32, kernel_size=3, stride=1, padding=0),
    nn.BatchNorm3d(32),
    nn.ReLU()
)

output = self.input_layer(x)

self.input_conv = nn.Conv3d(num_channels, 32, kernel_size=3, stride=1, padding=0)
self.input_bn = nn.BatchNorm3d(32)

output = F.relu(self.input_bn(self.input_conv(x)))