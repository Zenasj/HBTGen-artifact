import torch

with torch.jit.fuser("fuser2", force_fusion=True):
    fused_op = torch.jit.script(op)

with torch.jit.fuser("fuser2", force_fusion=True):
    c = torch.matmul(a, b)
    c = c.relu()