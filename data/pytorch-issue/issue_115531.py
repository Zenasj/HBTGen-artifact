import torch
import torch.nn as nn

x = torch.randn(3, 3, 7, 7)
p = torch.nn.FractionalMaxPool2d(kernel_size=4, output_ratio=0.8450560840300386, return_indices=False)
p(x)