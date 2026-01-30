import torch
from torchvision import models

model = models.resnet50()
input = torch.rand(1, 3, 224, 224)

# 1. loads SOL
# 2. loads the shared library that adds the memory allocator and function callbacks to PyTorch
# 3. adds the Python API to the torch module
import sol.pytorch

# Parses