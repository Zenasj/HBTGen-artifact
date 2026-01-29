# torch.rand(1, 3, 224, 224, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Use bilinear mode to align with ONNX requirements
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    
    def forward(self, x):
        return self.upsample(x)

def my_model_function():
    # Returns model with corrected bilinear upsample mode
    return MyModel()

def GetInput():
    # Generates random input tensor matching expected shape (B, C, H, W)
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

