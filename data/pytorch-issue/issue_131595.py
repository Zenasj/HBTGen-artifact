import torch

# Mode to emulate pytorch eager numerics for lower precision (fp16, bf16)
# Pytorch eager computes bf16/fp16 by upcasting inputs to fp32 and downcasting after
# For multiple, fused pointwise nodes, inductor will elide the intermediary upcasts and downcasts
# Typically this should be closer to fp64 ref numerics. However, it can be useful for debugging
# to emulate the eager numerics.

x = torch.rand([4], dtype=torch.bfloat16)
y = x + 1
z = y * 10

((x.to(torch.float32 + 1)) * 10).to(torch.bfloat16)