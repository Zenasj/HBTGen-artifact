import torch.nn as nn

import torch

a = torch.rand([128, 442368, 5], device="cuda", requires_grad=True)

mod = torch.nn.MaxPool1d(5)

out = mod(a)

out.backward(torch.ones_like(out))