import torch
import torch.nn as nn

conv = torch.nn.Conv2d(1, 1, kernel_size=3, dilation=2, stride=1)
tensor = torch.empty(1, 1, 4, 4)
conv(tensor).shape

conv = torch.nn.Conv2d(1, 1, kernel_size=3, dilation=2, stride=2)
tensor = torch.empty(1, 1, 4, 4)
conv(tensor).shape