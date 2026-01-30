import torch.nn as nn

py
import torch
from torch import nn

x = torch.randn(4, 32, 16, 16)
deconv = nn.LazyConvTranspose2d(64, 2, 2)
deconv(x, output_size=(33, 33)) # this throws the error in 2.2.2 but not in 2.1.0

py
import torch
from torch import nn

x = torch.randn(4, 32, 16, 16)
deconv = nn.LazyConvTranspose2d(64, 2, 2)
deconv(x, (33, 33)) # this throws an error in both versions