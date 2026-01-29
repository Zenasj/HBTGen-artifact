# torch.rand(B, 3, H, W, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, num_classes, compound_coef=3, ratios=None, scales=None, seg_classes=None, backbone_name=None, onnx_export=True):
        super(MyModel, self).__init__()
        # Dummy layers to mimic HybridNetsBackbone structure
        # Based on common backbone patterns and output requirements
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # Base feature extraction
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Intermediate features
        self.pool = nn.AdaptiveAvgPool2d(1)  # For classification output (output3)
        self.fc = nn.Linear(128, seg_classes)  # Segmentation classification
        # Output4 requires same spatial dimensions as input - using identity for demonstration
        self.identity = nn.Identity()  # Matches input shape for output4

    def forward(self, x):
        # Output1: (B, 64, H, W)
        out1 = self.conv1(x)
        # Output2: (B, 128, H, W)
        out2 = self.conv2(out1)
        # Output3: (B, seg_classes) via global pooling and FC
        pooled = self.pool(out2).view(out2.size(0), -1)
        out3 = self.fc(pooled)
        # Output4: Preserve input spatial dimensions (B, 3, H, W)
        out4 = self.identity(x)
        return out1, out2, out3, out4

def my_model_function():
    # Initialize with parameters inferred from the issue's code
    num_classes = 80  # Assumed from len(params.obj_list)
    seg_classes = 2   # Assumed from len(params.seg_list)
    ratios = (1.0, 2.0, 0.5)  # Example ratios from anchors_ratios
    scales = (1.0, 1.5, 2.0)  # Example scales from anchors_scales
    return MyModel(
        num_classes=num_classes,
        compound_coef=3,
        ratios=ratios,
        scales=scales,
        seg_classes=seg_classes,
        backbone_name=None,
        onnx_export=True
    )

def GetInput():
    # Generate dynamic input tensor matching expected dimensions
    B = 2  # Dynamic batch size
    C = 3  # Fixed channels
    H = 512  # Dynamic height
    W = 640  # Dynamic width
    return torch.randn(B, C, H, W, dtype=torch.float32)

