import torch

with torch.autocast("cpu", dtype=torch.bfloat16):
    assert torch.is_autocast_cpu_enabled()  # Works because the `dtype` was passed

with torch.autocast("cpu"):
    assert torch.is_autocast_cpu_enabled()  # Error!