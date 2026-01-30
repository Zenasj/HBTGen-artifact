import torch
import numpy as np

input = torch.tensor([np.inf], dtype=torch.complex128)
out = torch.atan(input)  # tensor([nan+nanj], dtype=torch.complex128)

input = torch.tensor([np.inf], dtype=torch.float64)
out = torch.atan(input)  # tensor([1.5708], dtype=torch.float64)

input = torch.tensor([np.inf], dtype=torch.complex128)
out = torch.atan(input)  # tensor([1.5708+0.j], dtype=torch.complex128)

input = torch.tensor([np.inf], dtype=torch.float64)
out = torch.atan(input)  # tensor([1.5708], dtype=torch.float64)