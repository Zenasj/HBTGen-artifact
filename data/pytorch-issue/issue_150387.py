# torch.rand(B, 9, 224, 224, dtype=torch.float32)
import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3 as DLV3


class MyModel(nn.Module):
    def __init__(self, in_channels=9, out_channels=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model = deeplabv3_resnet50(pretrained=True)
        self.process_model()  # Added to ensure layer modifications during initialization

    def process_model(self):
        # Modify input layer and output layers to match custom channels
        self.model.backbone.conv1 = nn.Conv2d(
            self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.model.classifier[-1] = nn.Conv2d(
            256, self.out_channels, kernel_size=1, stride=1
        )
        self.model.aux_classifier[-1] = nn.Conv2d(
            256, self.out_channels, kernel_size=1, stride=1
        )

    def freeze(self, flag=True):
        # Freeze/unfreeze layers as described in original code
        if flag:
            self.model.requires_grad_(False)
            self.model.backbone.conv1.requires_grad_(True)
            self.model.classifier[-1].requires_grad_(True)
            self.model.aux_classifier[-1].requires_grad_(True)
        else:
            self.model.requires_grad_(True)

    def forward(self, input):
        # Return full output during training (includes auxiliary), else only "out"
        return self.model(input) if self.training else self.model(input)["out"]


def my_model_function():
    # Return initialized model with default parameters (9 input channels, 1 output)
    return MyModel()


def GetInput():
    # Generate random tensor matching input shape (B=1 assumed for simplicity)
    return torch.rand(1, 9, 224, 224, dtype=torch.float32)

