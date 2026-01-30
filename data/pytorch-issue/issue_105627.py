import torch
import torch.nn as nn

p = nn.ZeroPad2d(padding=(1,2))
x = torch.randn(3,3,32,32)
y = p(x)
print(y.shape)

torch.Size([3, 3, 32, 35])

p = nn.ConstantPad3d(padding=1, value=0)
x = torch.randn(3,3,32,)
y = p(x)
print(y.shape)

torch.Size([5, 5, 34])