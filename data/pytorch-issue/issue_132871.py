import torch.nn as nn

import torch

query = (
        torch.empty(size=[2, 2, 49, 32], dtype=torch.bfloat16, device="cpu")
        .uniform_(-1, 1)
        .requires_grad_(True)
    )
key = (
        torch.empty(size=[2, 2, 49, 32], dtype=torch.bfloat16, device="cpu")
        .uniform_(-1, 1)
        .requires_grad_(True)
    )
value = (
        torch.empty(size=[2, 2, 49, 32], dtype=torch.bfloat16, device="cpu")
        .uniform_(-1, 1)
        .requires_grad_(True)
    )
breakpoint()
with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
    res = torch.nn.functional.scaled_dot_product_attention(query, key, value, None, 0.0, False)

res_grad = torch.empty_like(res, device="cpu").uniform_(-1, 1)

res.backward(res_grad, retain_graph=True)