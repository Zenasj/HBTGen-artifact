import torch.nn as nn

import torch
import torch.nn.functional as F

x = torch.randn(1, 256, 512, 512)
w = torch.randn(64, 256, 1, 1)

crop_ind = 99
conv_crop = torch.conv2d(x, w, None, [1, 1], [0, 0], [1, 1], 1)[:, :, :crop_ind, :crop_ind]
crop_conv = torch.conv2d(x[:, :, :crop_ind, :crop_ind], w, None, [1, 1], [0, 0], [1, 1], 1)
diff_sum = ((conv_crop - crop_conv) != 0).sum()