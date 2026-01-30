import torch
x = torch.randn(8, 224, 224, 3).permute(0, 3, 1, 2)
while True:
   x.cuda()

import torch
torch.set_num_threads(1)
x = torch.randn(8, 224, 224, 3).permute(0, 3, 1, 2)
while True:
   x.cuda()