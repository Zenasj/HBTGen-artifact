import torch.nn as nn

F.pixel_shuffle(x.cpu(), 2).to('mps')

import torch.nn.functional as F

class PixelShuffle(nn.Module):
    def __init__(self, upscale_factor):
        super(PixelShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # Calculate the new height and width
        new_height = height * self.upscale_factor
        new_width = width * self.upscale_factor

        # Calculate the channel size after shuffling
        new_channels = channels // (self.upscale_factor ** 2)

        # Reshape the input tensor
        x = x.view(batch_size, new_channels, self.upscale_factor, self.upscale_factor, height, width)

        # Transpose and flatten the tensor
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(batch_size, new_channels, new_height, new_width)

        return x