import torch.nn as nn

import torch
import torch.nn.functional as F

# works (cpu)
x = torch.randn(1, 1)
F.pad(x, (7, 0), mode="constant", value=0)

# does not work (mps)
x = torch.randn(1, 1).to("mps")
F.pad(x, (7, 0), mode="constant", value=0)