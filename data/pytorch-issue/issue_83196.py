# torch.rand(B, C, H, W, dtype=torch.float32)  # Add a comment line at the top with the inferred input shape

import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, inFM, outFM, kSize, outSize, stride=1, padding=0, output_padding=0):
        super(MyModel, self).__init__()
        self.pixel_shuffle_layers = nn.Sequential(
            nn.Conv2d(inFM, 4 * outFM, kSize, stride=1, padding=padding),
            nn.PixelShuffle(stride),
            nn.LayerNorm([outFM, outSize, outSize]),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv_transpose_layers = nn.Sequential(
            nn.ConvTranspose2d(inFM, outFM, kSize, stride=stride, padding=padding, output_padding=output_padding),
            nn.LayerNorm([outFM, outSize, outSize]),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        pixel_shuffle_output = self.pixel_shuffle_layers(x)
        conv_transpose_output = self.conv_transpose_layers(x)
        return pixel_shuffle_output, conv_transpose_output

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel(inFM=512, outFM=128, kSize=3, outSize=64, stride=2, padding=1, output_padding=1)

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    B, C, H, W = 64, 512, 32, 32
    return torch.rand(B, C, H, W, dtype=torch.float32)

