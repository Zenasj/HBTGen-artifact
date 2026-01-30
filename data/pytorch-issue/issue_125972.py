import torch

xla_device = xm.xla_device()
xla_tensor_0 = torch.tensor(42, dtype=torch.uint32).to(xla_device)