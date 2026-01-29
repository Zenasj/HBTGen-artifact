# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
from torchvision.models import MobileNetV2

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model_fp32 = MobileNetV2()  # FP32 model
        self.model_fp16 = MobileNetV2().half()  # FP16 model

    def forward(self, x):
        # FP32 path: run features[:7] in FP32
        features_fp32 = self.model_fp32.features[:7](x.float())
        # FP16 path: convert input to FP16 and run features[:7]
        features_fp16 = self.model_fp16.features[:7](x.half())
        
        # Check for NaN in FP16 features and absence in FP32
        has_nan_fp32 = torch.isnan(features_fp32).any().item()
        has_nan_fp16 = torch.isnan(features_fp16).any().item()
        
        # Return True (1) if FP16 has NaN but FP32 does not
        condition = (not has_nan_fp32) and has_nan_fp16
        return torch.tensor([condition], dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    # Return a random input tensor matching MobileNetV2's expected input shape
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

