import torch.nn as nn

import torch

device = torch.device("mps")
a = torch.rand(1, 1, 32, 32)
b = torch.rand(1, 1, 32, 32)
a_mps = a.to(device)
b_mps = b.to(device)

lossfn_no_reduction = torch.nn.BCELoss(reduction="none")
lossfn_w_reduction = torch.nn.BCELoss(reduction="mean")

# OK
lossfn_no_reduction(a_mps, b_mps)

# OK
lossfn_w_reduction(a, b)

# error
lossfn_w_reduction(a_mps, b_mps)