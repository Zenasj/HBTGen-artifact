import torch
import torch.nn as nn
import numpy as np

x = np.ones(200000) * (1 + 1j)  # won't crash with 100000
x_t = torch.tensor([[x]])
vec = torch.nn.MaxPool1d(100)(x_t.real)