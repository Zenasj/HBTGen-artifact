import math
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

in_channels = 120
groups = 2
kernel = (3, 8)
m = nn.Conv2d(in_channels=in_channels, groups=groups,
              out_channels=100, kernel_size=kernel)

k = math.sqrt(groups / (in_channels * math.prod(kernel)))
print(f"k: {k:0.6f}")

print(f"min weight: {m.weight.min().item():0.6f}")
print(f"max weight: {m.weight.max().item():0.6f}")

_ = plt.hist(m.weight.detach().numpy().ravel())