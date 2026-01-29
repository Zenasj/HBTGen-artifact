# torch.rand(1, 64, 62, dtype=torch.float32)  # Add a comment line at the top with the inferred input shape

import torch
import torch.nn as nn
import torch.nn.functional as F

class VggishConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VggishConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=(2, 2))
        return x

class MyModel(nn.Module):
    def __init__(self, classes_num):
        super(MyModel, self).__init__()
        self.conv_block1 = VggishConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = VggishConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = VggishConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = VggishConvBlock(in_channels=256, out_channels=512)
        self.fc_final = nn.Linear(512, classes_num, bias=True)

    def forward(self, input):
        (_, seq_len, mel_bins) = input.shape
        x = input.view(-1, 1, seq_len, mel_bins)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = F.max_pool2d(x, kernel_size=x.shape[2:])
        x = x.view(x.shape[0:2])
        x = F.log_softmax(self.fc_final(x), dim=-1)
        return x

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel(classes_num=41)

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.FloatTensor(1, 64, 62)

