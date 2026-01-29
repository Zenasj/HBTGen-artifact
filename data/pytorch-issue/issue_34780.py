# torch.rand(1, 3, 512, 512, dtype=torch.float32)  # Add a comment line at the top with the inferred input shape

import torch
import torch.nn as nn
import torch.nn.functional as F

class BiSeNet(nn.Module):
    def __init__(self, num_classes):
        super(BiSeNet, self).__init__()
        # Define the base model (ResNet18 in this case)
        self.base = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # Define the average pooling layer with a fixed kernel size
        self.avg_pool2d = nn.AvgPool2d(kernel_size=(16, 16), ceil_mode=False)

        # Define the final convolutional layer
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        feat = self.base(x)
        feat_shape = [int(s) for s in feat.shape[2:]]
        feat = F.avg_pool2d(feat, feat_shape)
        out = self.final_conv(feat)
        return out

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return BiSeNet(19)

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(1, 3, 512, 512, dtype=torch.float32).cuda()

