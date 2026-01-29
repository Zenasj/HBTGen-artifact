# torch.rand(B, 4096, dtype=torch.float32)
import torch
from torch import nn
import torch.ao.quantization as quant

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Base linear layer with dynamic quantization applied
        self.original_layer = nn.Linear(4096, 4096, bias=False)
        # Wrap with quantized version as submodule
        self.quantized_model = quant.quantize_dynamic(self.original_layer, {nn.Linear})

    def forward(self, x):
        # Forward through quantized model
        return self.quantized_model(x)

def my_model_function():
    # Returns quantized model instance
    return MyModel()

def GetInput():
    # Matches input shape (BATCH, 4096) with float32 dtype
    # Example uses BATCH sizes 32 and 16, so B=32 is safe default
    return torch.rand(32, 4096, dtype=torch.float32)

