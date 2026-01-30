import torch.nn as nn

import torch

# Tensor Parameters
BATCH = 10
CHANNEL = 16
HEIGHT = 2048
WIDTH = 2048
DTYPE = torch.float32
DEVICE = "cpu"

# Pooling Parameters
KERNEL_SIZE = 4
STRIDE = 1
PADDING = 1
DILATION = 2
CEIL_MODE = True

blob = torch.randn(BATCH, CHANNEL, HEIGHT, WIDTH, dtype=DTYPE, device=DEVICE)

model = torch.nn.MaxPool2d(KERNEL_SIZE, STRIDE, PADDING, DILATION, False, CEIL_MODE)
model(blob.to_mkldnn())