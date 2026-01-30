import torch
import torch_xla
import torch_xla.core.xla_model as xm
from torch_xla.amp import autocast, GradScaler

device = xm.xla_device()

a_float32 = torch.rand((8, 8), device=device)
b_float32 = torch.rand((8, 8), device=device)

with autocast():
    e_float16 = torch.mm(a_float32, b_float32)
    print(e_float16.dtype)