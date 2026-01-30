import torch
import torch._dynamo.config

torch._dynamo.config.capture_scalar_outputs = True

@torch.compile(fullgraph=True)
def f(x):
    device = x.device
    s, s2 = x.tolist()
    g = torch.randn(s, device=device)
    g2 = torch.randn(s2, device=device)
    return torch.ops.aten.cat.default([g, g, g2])

f(torch.tensor([4, 6], device='cuda'))