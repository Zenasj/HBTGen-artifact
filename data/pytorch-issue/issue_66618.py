import torch
a = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
torch.autograd.gradcheck(lambda x: torch.linalg.pinv(x, rcond=torch.tensor(1e-15)), [a])
# RuntimeError: derivative for aten::linalg_pinv is not implemented

import torch
a = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
rcond = torch.tensor(1e-15, dtype=torch.float64, requires_grad=True)
torch.autograd.gradcheck(lambda rcond: torch.linalg.pinv(a, rcond=rcond), [rcond])
# True in 1.9.0 and errors on master