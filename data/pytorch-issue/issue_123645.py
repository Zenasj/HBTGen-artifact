import torch

v1 = torch.randn(354298880)
v1.norm()
# Output: tensor(17666.5586)
v1.double().norm()
# Output: tensor(18823.4125, dtype=torch.float64)
v1.half().norm()
# Output: tensor(18752., dtype=torch.float16)