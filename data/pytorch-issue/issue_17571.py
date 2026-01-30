import torch
import torch.nn as nn

torch.jit.trace(nn.utils.spectral_norm(nn.Linear(8,8)), torch.rand(4,8))