import torch.nn as nn

import torch

def fn(x):
    with torch.nn.attention.sdpa_kernel(
        # pyre-fixme[16]: Module `torch.nn.attention` has no attribute `SDPBackend`.
        [torch.nn.attention.SDPBackend.FLASH_ATTENTION]
    ):
        output = torch.nn.functional.scaled_dot_product_attention(
            x, x, x
        ).to(torch.float32)
    return output

x = torch.randn(10, 4, 128, 16).to(dtype=torch.float16)

torch.compile(fn)(x)