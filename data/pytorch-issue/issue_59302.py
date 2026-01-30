import torch
import torch.nn as nn
from torch.nn.utils import weight_norm, spectral_norm
conv_spectral = spectral_norm(nn.Conv2d(3, 16, 3))
torch.jit.script(conv_spectral)