import torch
import torch.nn as nn

torch.nn.GRU(5, 1).to(torch.complex64)(torch.zeros(3, 4, 5, dtype=torch.complex64))