import torch.nn as nn

import torch

li = torch.nn.Linear(1000,2000)
li.eval()

ex = torch.rand(2,1000)
ex3 = ex[:1]

out = li(ex)
out3 = li(ex3)

print(out[0,0].item(), out3[0,0].item())
# 0.6420710682868958 0.6420711278915405