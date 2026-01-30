import torch
import torch_xla.core.xla_model as xm

device = xm.xla_device()

def foo():
    return torch.tensor([0.0], device=device)

compiled = torch.compile(backend="openxla")(foo)
tensor = compiled()