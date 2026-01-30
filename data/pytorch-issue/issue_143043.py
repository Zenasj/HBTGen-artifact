import torch.nn as nn

import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
x = torch.rand(1, 3, 64, 64, device=device)  # Random tensor
conv = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1).to(device)
y = conv(x)
print(y.shape)