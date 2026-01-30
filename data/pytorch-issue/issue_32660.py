import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class ModulatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, kernel_size=3, stride=1, padding=1, dilation=1,
                 noisy=True, randomize_noise=True, up=False, demodulize=True, gain=1, lrmul=1):
        super(ModulatedConv2d, self).__init__()
        assert kernel_size >= 1 and kernel_size % 2 == 1
        self.noisy = noisy
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.randomize_noise = randomize_noise
        self.up = up
        self.demodulize = demodulize
        self.lrmul = lrmul

        # Get weight.
        fan_in = in_channels * kernel_size * kernel_size
        self.runtime_coef = gain / math.sqrt(fan_in) * math.sqrt(lrmul)
        self.weight = Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) / math.sqrt(lrmul), requires_grad=True) # [OIkk]

        # Get bias.
        self.bias = Parameter(torch.zeros(1, out_channels, 1, 1), requires_grad=True)

        # Modulate layer.
        self.mod = ScaleLinear(hidden_channels, in_channels, bias=True) # [BI] Transform incoming W to style.

        # Noise scale.
        if noisy:
            self.noise_scale = Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x, y, noise=None):
        w = self.weight * self.runtime_coef
        ww = w[np.newaxis] # [BOIkk] Introduce minibatch dimension.

        # Modulate.
        s = self.mod(y) + 1 # [BI] Add bias (initially 1).
        ww = ww * s[:, np.newaxis, :, np.newaxis, np.newaxis] # [BOIkk] Scale input feature maps.

        # Demodulate.
        if self.demodulize:
            d = torch.rsqrt(ww.pow(2).sum(dim=(2,3,4), keepdim=True) + 1e-8) # [BOIkk] Scaling factor.
            ww = ww * d # [BOIkk] Scale output feature maps.

        # Reshape/scale input.
        B = y.size(0)
        x = x.view(1, -1, *x.shape[2:]) # Fused [BIhw] => reshape minibatch to convolution groups [1(BI)hw].
        w = ww.view(-1, *ww.shape[2:]) # [(BO)Ikk]

        # Convolution with optional up/downsampling.
        if self.up: x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = F.conv2d(x, w, None, self.stride, self.padding, self.dilation, groups=B) # [1(BO)hw]

        # Reshape/scale output.
        x = x.view(B, -1, *x.shape[2:]) # [BOhw]

        # Apply noise and bias
        if self.noisy:
            if self.randomize_noise: noise = x.new_empty(B, 1, *x.shape[2:]).normal_()
            x += noise * self.noise_scale
        x += self.bias * self.lrmul
        return x