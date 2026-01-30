import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalPoolingLayer(nn.Module):
    def __init__(self, channels):
        super(GlobalPoolingLayer, self).__init__()
        self.CHANNELS = channels

    def forward(self, x):
        # Mean of each channel
        mean_pool = F.adaptive_avg_pool2d(x, 1)  # b x c
        mean_pool = mean_pool.view(-1, self.CHANNELS)

        # Maximum of each channel
        max_pool = F.adaptive_max_pool2d(x, 1)  # b x c
        max_pool = max_pool.view(-1, self.CHANNELS)

        # Concatenate the results
        return torch.cat((mean_pool, max_pool), dim=1)  # b x (2c)