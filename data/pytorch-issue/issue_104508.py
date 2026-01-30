import torch.nn as nn

m = nn.ConstantPad3d(padding=1, value=1)
m(torch.rand([1,2]))

l = nn.ConstantPad2d(padding=1, value=1)
l(torch.rand([2]))