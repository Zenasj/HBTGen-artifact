import torch
import torch.nn as nn
from typing import Optional

# torch.rand(B, 1, 1, 1, dtype=torch.float32)
class MyModel(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lora_layer: Optional[nn.Module] = None

    def set_lora_layer(self, lora_layer: Optional[nn.Module]):
        self.lora_layer = lora_layer

    def forward(self, x):
        base_output = super().forward(x)
        if self.lora_layer is not None:
            return base_output + self.lora_layer(x)
        return base_output

def my_model_function():
    model = MyModel(1, 1)  # Matches original Linear(1,1) initialization
    nn.init.ones_(model.weight)       # Initial weight = 1
    nn.init.zeros_(model.bias)        # Initial bias = 0
    model.set_lora_layer(None)        # Start without LoRA
    return model

def GetInput():
    return torch.rand(1, 1, 1, dtype=torch.float32)  # Matches original test input shape

