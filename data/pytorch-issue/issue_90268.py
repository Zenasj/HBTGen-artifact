# torch.rand(1, 3, 224, 224, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Simplified structure to mimic Swin-V2-T's problematic layer
        self.conv = nn.Conv2d(3, 64, kernel_size=3)
        
    def forward(self, x):
        x = self.conv(x)
        # The following line uses 'zero_', which triggers the ONNX export error
        x.zero_()  # aten::zero_() unsupported in opset 14
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

