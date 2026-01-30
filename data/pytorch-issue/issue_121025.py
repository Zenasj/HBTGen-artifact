import torch.nn as nn

loss_fn = nn.MSELoss()
loss_fn(gt_corners, corners)