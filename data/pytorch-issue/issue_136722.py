import torch.nn as nn

import torch
torch.ao.nn.quantized.ConvTranspose1d(16, 64, kernel_size=3, stride=0) # crash
torch.ao.nn.quantized.ConvTranspose2d(16, 64, kernel_size=3, stride=0) # crash
torch.ao.nn.quantized.ConvTranspose3d(16, 64, kernel_size=3, stride=0) # crash