# torch.rand(B=2, C=3, H=800, W=800, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Simulated RetinaNet backbone (ResNet-like structure)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # Simplified ResNet blocks (assuming 34/50 layers)
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
        )
        # Mock classification and regression heads
        self.cls_head = nn.Conv2d(64, 9*80, kernel_size=3, padding=1)  # 9 anchors * 80 classes
        self.reg_head = nn.Conv2d(64, 9*4, kernel_size=3, padding=1)  # 9 anchors * 4 coords
        
    def forward(self, x):
        x = self.backbone(x)
        cls_logits = self.cls_head(x).permute(0, 2, 3, 1).contiguous()  # Shape: (B, H, W, C)
        bbox_regression = self.reg_head(x).permute(0, 2, 3, 1).contiguous()
        # Avoid control flow and ensure outputs are pure tensors
        return [cls_logits.view(cls_logits.shape[0], -1), 
                bbox_regression.view(bbox_regression.shape[0], -1)]

def my_model_function():
    model = MyModel()
    # Initialize weights (mock initialization)
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean=0, std=0.01)
    return model

def GetInput():
    return torch.randn(2, 3, 800, 800, dtype=torch.float32)

