import torch.nn.functional as F

y = F.avg_pool2d(y, kernel_size=y.size()
                             [2:]).view(y.size(0), -1)

y = y.flatten(start_dim=2).mean(dim=2)