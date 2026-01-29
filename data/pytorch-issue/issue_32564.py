# torch.rand(B, 3, 224, 224, dtype=torch.float32)  # Inferred input shape based on common use and loss function processing
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, dce_nChannels=1):
        super(MyModel, self).__init__()
        self.dce_nChannels = dce_nChannels
        nChannels = self.dce_nChannels + 2  # Matches loss function's nChannels derivation
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        # Auxiliary branch for CELoss
        self.aux_conv = nn.Conv2d(32, nChannels, kernel_size=1)
        # Main branch for MaskedMSE
        self.main_conv = nn.Conv2d(32, 1, kernel_size=1)  # Output for depth regression

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        pred_aux = self.aux_conv(x)  # For auxiliary classification loss
        main_out = self.main_conv(x)  # For depth regression loss
        return pred_aux, main_out

def my_model_function():
    # Initialize with default parameters matching the loss functions' expected structure
    return MyModel(dce_nChannels=1)

def GetInput():
    # Returns a tensor matching the input expected by MyModel
    B, C, H, W = 1, 3, 224, 224  # Batch size 1 assumed for minimal reproduction
    return torch.rand(B, C, H, W, dtype=torch.float32)

