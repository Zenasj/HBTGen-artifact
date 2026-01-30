import torch.nn as nn

import torch

# Create model.
module = torch.nn.Conv3d(
    in_channels=4,
    out_channels=4,
    kernel_size=3,
    bias=True,
)
module = torch.nn.utils.parametrizations.weight_norm(module)  # <-- works if commented out
module.eval()

module = torch.compile(module)

# Run it with some random input.
input = torch.rand((1, 4, 3, 3, 3), dtype=torch.float32)

with torch.no_grad():
    module(input)