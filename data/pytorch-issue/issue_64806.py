import torch.nn as nn

# code in Resnet
self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

################################################

# code in my module
pool_scales = (1, 2, 3, 6)
self.ppm_pooling = []
for scale in pool_scales:
    self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
self.ppm_pooling = nn.ModuleList(self.ppm_pooling)