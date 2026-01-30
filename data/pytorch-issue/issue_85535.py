import torch.nn as nn

import torch
import functorch

lr = torch.nn.LeakyReLU(0.2, inplace=True)
p = torch.randn(4, 4)
t = torch.randn(4, 4)

functorch.jvp(lr, (p,), (t,))