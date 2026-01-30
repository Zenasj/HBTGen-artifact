import torch.nn.functional as F

x = F.avg_pool2d(x, x.shape[2:])

x_shape = [int(s) for s in x.shape[2:]]
x = F.avg_pool2d(x, x_shape)