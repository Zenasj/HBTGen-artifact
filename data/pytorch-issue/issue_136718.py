import torch.nn as nn

import torch
torch.ao.nn.quantized.Conv1d(16, 64, kernel_size=3, stride=0) # crash
torch.ao.nn.quantized.Conv2d(16, 64, kernel_size=3, stride=0) # crash
torch.ao.nn.quantized.Conv3d(16, 64, kernel_size=3, stride=0) # crash