import torchvision

import torch
import torch.nn as nn
import numpy as np
import random
import os


def seed_fixed(seed=2025):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_fixed()

from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation import DeepLabV3 as DLV3


class DeepLabV3(nn.Module):
    def __init__(self, in_channels=9, out_channels=1):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model = deeplabv3_resnet50(pretrained=True)
        self.model: DLV3

    def process_model(self):
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
        if flag:
            self.model.requires_grad_(False)
            self.model.backbone.conv1.requires_grad_(True)
            self.model.classifier[-1].requires_grad_(True)
            self.model.aux_classifier[-1].requires_grad_(True)
        else:
            self.model.requires_grad_(True)

    def forward(self, input):
        if self.training:
            return self.model(input)
        else:
            return self.model(input)["out"]


if __name__ == "__main__":
    pass