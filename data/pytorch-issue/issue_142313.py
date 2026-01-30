import torch.nn as nn

import torch
from torch.nn.attention.flex_attention import flex_attention


@torch.compile(backend="aot_eager")
def fwd_bwd(x: torch.Tensor):
    flex_attention(x, x, x).sum().backward()


v = torch.zeros(1, 1, 768, 64, dtype=torch.bfloat16, device="cuda", requires_grad=True)
with torch._dynamo.compiled_autograd._enable(torch.compile(backend="aot_eager")):
    fwd_bwd(v)