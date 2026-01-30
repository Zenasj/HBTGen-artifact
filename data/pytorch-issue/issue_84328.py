import torch
import torch.nn as nn

myfc = nn.Linear(512 * 7 * 6, 512).cuda()  # should move to cuda
x = torch.randn(5, 512 * 7 * 6 + 1).cuda()  # -1 is not ok, +1 is ok
out = myfc(x)
print(out.shape)