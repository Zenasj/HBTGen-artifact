import torch.nn as nn

class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        super(SegmentationHead, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        self.activation = Activation(activation)
    def forward(self, x):
        x = self.conv2d(x)
        x = self.upsampling(x)
        x = self.activation(x)
        return x