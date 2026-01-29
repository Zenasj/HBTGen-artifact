# torch.rand(B, 10, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, enabled=True):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.enabled = enabled
    
    def forward(self, x):
        if self.enabled and torch.cuda.is_bf16_supported():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                return self.linear(x)
        else:
            return self.linear(x)

def my_model_function():
    return MyModel()

def GetInput():
    B = 2  # Example batch size
    return torch.rand(B, 10, dtype=torch.float32)

