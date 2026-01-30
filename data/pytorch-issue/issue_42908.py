import torch

a = torch.Tensor().ndim  # mypy says: `error: "Tensor" has no attribute "ndim"`