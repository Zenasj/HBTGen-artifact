import torch

inp = torch.randn(1, device='cuda', dtype=torch.half)
with torch.autocast('cuda', dtype=torch.float16):
    torch.linalg.vector_norm(inp)