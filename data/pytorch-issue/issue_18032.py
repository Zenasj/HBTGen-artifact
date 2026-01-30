import torch.nn as nn

import torch
import torch.nn.functional as F

x = torch.Tensor(torch.Size([64, 2, 3])).to(torch.device("cuda"))
grid = F.affine_grid(x, torch.Size([64, 1, 28, 28]))