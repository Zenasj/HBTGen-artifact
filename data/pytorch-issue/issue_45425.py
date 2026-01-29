# torch.rand(B, 3, 256, 256, dtype=torch.float32)
import torch
import torch.nn as nn
from torchvision import models

class ConvRelu(nn.Module):
    def __init__(self, in_depth, out_depth):
        super(ConvRelu, self).__init__()
        self.conv = nn.Conv2d(in_depth, out_depth, kernel_size=3, stride=1, padding=1)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_depth, middle_depth, out_depth):
        super(DecoderBlock, self).__init__()
        self.conv_relu = ConvRelu(in_depth, middle_depth)
        self.conv_transpose = nn.ConvTranspose2d(middle_depth, out_depth, kernel_size=4, stride=2, padding=1)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_relu(x)
        x = self.conv_transpose(x)
        x = self.activation(x)
        return x

class MyModel(nn.Module):
    def __init__(self, num_classes):
        super(MyModel, self).__init__()
        self.encoder = models.resnet101(pretrained=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Sequential(
            self.encoder.conv1,
            self.encoder.bn1,
            self.encoder.relu,
            self.pool
        )
        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3  # Fixed assignment
        self.conv5 = self.encoder.layer4  # Added missing layer

        self.center = DecoderBlock(2048, 512, 256)
        self.dec5 = DecoderBlock(2048 + 256, 512, 256)
        self.dec4 = DecoderBlock(1024 + 256, 512, 256)
        self.dec3 = DecoderBlock(512 + 256, 256, 64)
        self.dec2 = DecoderBlock(256 + 64, 128, 128)
        self.dec1 = DecoderBlock(128, 128, 32)
        self.dec0 = ConvRelu(32, 32)
        self.final = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)  # Now valid

        pool = self.pool(conv5)
        center = self.center(pool)

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))  # Corrected decoder usage
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))  # Corrected decoder usage
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)

        return self.final(dec0)

def my_model_function():
    # Assuming num_classes=1 for binary segmentation (common in satellite segmentation tasks)
    return MyModel(num_classes=1)

def GetInput():
    # Matches input shape expected by MyModel's first layer (3 channels, 256x256)
    return torch.rand(4, 3, 256, 256, dtype=torch.float32)

