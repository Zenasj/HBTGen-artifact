import torch
import torch.nn as nn

x = torch.rand((31, 64, 512, 512)).cuda().to(memory_format=torch.channels_last)
torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest').shape

x = torch.rand((32, 64, 512, 512)).cuda().to(memory_format=torch.channels_last)
torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest').shape

x = torch.rand((32, 64, 512, 512)).cuda()
torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest').shape