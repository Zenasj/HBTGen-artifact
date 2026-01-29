# torch.rand(B, 3, 224, 224, dtype=torch.float32)
import torch
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = QuantStub()  # Quantization stub for input
        self.dequant = DeQuantStub()  # Dequantization stub for output
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=16, stride=16),  # Matches DeiT Tiny patch size 16
            nn.LayerNorm(192),  # Norm after patch embedding (patch_embed_norm in error)
        )
        self.fc = nn.Linear(192, 10)  # Dummy output layer

    def forward(self, x):
        x = self.quant(x)  # Quantize input
        x = self.patch_embed[0](x)  # Conv2d to extract patches
        # Simulate potential problematic concatenation path
        # Split into two parts and concatenate (axis 1 corresponds to channels here)
        x = torch.cat([x[:, :96], x[:, 96:]], dim=1)  # Split along channels (axis 1)
        x = self.patch_embed[1](x.mean((2, 3)))  # Simplified norm + flatten for FC
        x = self.fc(x)
        return self.dequant(x)  # Dequantize output

def my_model_function():
    model = MyModel()
    model.eval()  # Ensure in inference mode for quantization
    return model

def GetInput():
    # Input shape for DeiT Tiny (224x224 RGB images)
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

