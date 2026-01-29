# torch.rand(B, 3, 224, 224, dtype=torch.float32)
import torch
from torch import nn
from dataclasses import dataclass

@dataclass(frozen=True)
class _ActQuantizer:
    target_dtype: torch.dtype
    quant_min: int = -127
    dynamic: bool = True

    def dynamic_quantize(self, input):
        # Stub implementation for dynamic quantization
        return input  # Actual logic would involve quantization steps

    def static_quantize(self, input):
        # Stub implementation for static quantization
        return input  # Actual logic would involve quantization steps

    def __call__(self, *args):
        if self.dynamic:
            return self.dynamic_quantize(*args)
        else:
            return self.static_quantize(*args)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.quantizer = _ActQuantizer(target_dtype=torch.int, dynamic=True)
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.fc = nn.Linear(16 * 224 * 224, 10)

    def forward(self, x):
        x = self.quantizer(x)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

