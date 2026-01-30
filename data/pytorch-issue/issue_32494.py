import torch
from torch.quasirandom import SobolEngine

torch.set_default_tensor_type(torch.cuda.FloatTensor)
se = SobolEngine(3)  # crashes w/o any error message here