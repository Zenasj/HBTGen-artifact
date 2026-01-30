import torch.nn as nn

import torch
m = torch.nn.Upsample(size=(2, 2), recompute_scale_factor=True)
print(m)