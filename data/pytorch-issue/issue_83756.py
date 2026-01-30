import torch
from torch import linalg

linalg.cross(torch.randn(2, 3), torch.randn(5, 2, 3), dim=-1) # Dimension -1 does not have size 3
linalg.cross(torch.randn(2, 3), torch.randn(5, 2, 3), dim=2)  # Dimension out of range