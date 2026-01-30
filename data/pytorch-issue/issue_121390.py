import torch
import torch.nn as nn

mode = "nearest"
input = torch.randn(1, 1, 4, 2)
output = torch.nn.functional.interpolate(input, scale_factor=(2.05, 3.15), mode=mode)
print(input)
print(output)