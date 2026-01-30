import torch.nn as nn

import torch
from torch import nn, functional as F

DIM2_SIZE = 1

if __name__ == "__main__":
    print(f"Torch version: {torch.__version__}")
    torch.manual_seed(42)
    inputs = torch.randn(16, DIM2_SIZE, dtype=torch.float32)

    layer_norm = nn.LayerNorm(DIM2_SIZE)

    outputs = layer_norm(inputs)

    print(f"layer norm output:\n{outputs}")

import torch
from torch import nn, functional as F

DIM2_SIZE = 1

if __name__ == "__main__":
    print(f"Torch version: {torch.__version__}")
    torch.manual_seed(42)
    inputs = torch.randn(16, DIM2_SIZE, dtype=torch.float32)
    print(f"inputs:\n{inputs}")

    layer_norm = nn.LayerNorm(DIM2_SIZE)

    outputs = layer_norm(inputs)

    print(f"layer norm output:\n{outputs}")