import torch
import torch.nn as nn

x = torch.randn(3, 64, 8, 12)
p = nn.ReflectionPad2d(padding=9)
p(x)