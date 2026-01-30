import torch.nn as nn

import torch

# For 1d
torch.nn.functional.adaptive_avg_pool1d(torch.randn([2,2,2]), 9132760301568586890)

# For 2d
torch.nn.functional.adaptive_avg_pool2d(torch.randn([2,2,2,2]), 9132760301568586890)

# For 3d
torch.nn.functional.adaptive_avg_pool3d(torch.randn([2,2,2,2]), 9132760301568586890)