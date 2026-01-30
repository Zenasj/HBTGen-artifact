import torch.nn as nn

import torch

@torch.compile
def foo(a: torch.Tensor) -> torch.Tensor:
    num_norm_dims = 1
    norm_shape = a.shape[a.ndim - num_norm_dims:]
    layernorm = torch.nn.LayerNorm(
        norm_shape, elementwise_affine=True
    )
    return layernorm(a)

device = torch.device("cuda")
x = torch.randn(512, 128).to(device)
result = foo(x)