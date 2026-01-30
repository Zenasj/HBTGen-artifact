import torch

@torch.jit.script
def fn(m):
    # type: (Tensor) -> Device
    return m.device

@torch.jit.script
def fn(m: Tensor) -> torch.Device:
    return m.device