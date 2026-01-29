# torch.rand(B, 10, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Quantized module structure (simplified for demonstration)
        self.quantized_layer = nn.Linear(10, 5)
        # Non-quantized module for comparison (from ModuleNoTensors example)
        self.non_quantized_part = nn.Identity()  # Placeholder for ModuleNoTensors logic

    def forward(self, x):
        # Run both paths and return outputs for comparison
        quantized_out = self.quantized_layer(x)
        non_quantized_out = self.non_quantized_part(x)
        return quantized_out, non_quantized_out

    def __getstate__(self):
        # Triggers error during torch.save due to PR's serialization check
        return {'quantized': self.quantized_layer, 'non_quantized': self.non_quantized_part}

def my_model_function():
    model = MyModel()
    # Initialize weights (simplified)
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
    return model

def GetInput():
    # Input shape matching the model's expected input
    return torch.rand(2, 10, dtype=torch.float32)

