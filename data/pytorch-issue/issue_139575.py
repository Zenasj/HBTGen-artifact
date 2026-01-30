import torch

param_group["lr"] = torch.tensor(0.0) if isinstance(param_group["lr"], torch.Tensor) else 0.0