import torch

if not is_available():
    raise RuntimeError("torch.distributed is not available")