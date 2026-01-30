import torch_xla.core.xla_model as xm
from torch.amp import autocast

with autocast("cuda", dtype=torch.bfloat16):
    x = torch.tensor([5.], device=xm.xla_device())