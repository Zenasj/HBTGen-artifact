import torch
import torch.nn as nn
m = nn.Conv2d(4, 4, kernel_size=3, groups=2)
i = torch.randn(2, 4, 6, 6, dtype=torch.float, requires_grad=True)
y = m(i)

y = m(i)