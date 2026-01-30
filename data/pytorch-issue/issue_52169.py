import torch

@torch.jit.script
def fake_q(x, scale, dtype_min: float, dtype_max: float):
    f = x / scale
    f = torch.round(f)
    f = f.clamp(dtype_min, dtype_max)
    f = f * scale
    return f