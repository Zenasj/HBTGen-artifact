# torch.rand(B, 3, 416, 416, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.net_info = {"height": 416}  # Matches model.net_info["height"] usage
        # Simplified Darknet backbone (stride and dimensionality adjusted for YOLOv3)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
        )
        # Detection head (1x1 conv for YOLO-style outputs)
        self.detection_head = nn.Conv2d(128, 255, kernel_size=1)  # 3 anchors × (5+80 classes)

    def forward(self, x, cuda=False):
        # Forward path mimics YOLOv3 structure
        x = self.backbone(x)
        return self.detection_head(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 416, 416, dtype=torch.float32)

