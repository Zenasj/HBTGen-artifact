import torch

import T_B as torch

torch.p2()  # IDE can detect `p2`
torch.p1    # IDE cannot detect `p1`

for name in dir(_C._VariableFunctions):
    globals()[name] = getattr(_C._VariableFunctions, name)