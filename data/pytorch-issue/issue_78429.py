import torch.nn as nn

import torch

tensor = torch.randn((8, 3, 224, 224), device="mps")
norm = torch.nn.LayerNorm(224).to("mps")
output = norm(tensor)  # the error is thrown here