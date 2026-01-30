import torch.nn as nn
import torch.nn.functional as F
import torchvision

import torch
from torch.nn.functional import conv2d

@torch.compile(backend="inductor", dynamic=True)
def fn(image):
    num_channels = image.shape[-3]
    kernel = torch.rand(num_channels, 1, 3, 3)
    return conv2d(image, kernel, groups=num_channels)

from torchvision.transforms.v2 import functional as F

cfn = torch.compile(backend="inductor", dynamic=True)(F.gaussian_blur_image)
cfn(torch.rand(3, 16, 16), [3, 3])

cfn = torch.compile(backend="inductor", dynamic=True)(F.adjust_sharpness_image)
cfn(torch.rand(3, 16, 16), 0.5)