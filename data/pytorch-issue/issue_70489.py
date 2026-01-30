import torch.nn as nn

import torch
arg_class = torch.nn.RReLU(0.1, 0.3).cuda()
tensor = torch.rand([0], dtype=torch.float32).cuda()
print(tensor)
arg_class(tensor)
# floating point exception (core dumped)