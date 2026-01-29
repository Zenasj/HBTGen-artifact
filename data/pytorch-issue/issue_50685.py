# torch.rand(B, 3, 736, 1312, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # InceptionResNetv2 backbone (simplified structure)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # InceptionA blocks (hypothetical)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            # FPN layers (simplified)
            nn.Conv2d(128, 256, kernel_size=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        )
        # Feature Pyramid Network (FPN) head
        self.fpn_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3, kernel_size=1),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.fpn_head(x)
        return x

def my_model_function():
    # Initialize model with default parameters (weights loading omitted due to missing checkpoint)
    model = MyModel()
    return model

def GetInput():
    # Input tensor matching the exported dummy input dimensions
    return torch.randn(1, 3, 736, 1312, dtype=torch.float32)

