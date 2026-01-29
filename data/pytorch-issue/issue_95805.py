# torch.rand(B, S, E, dtype=torch.float32)  # Example shape: (2, 16, 768)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Simulate frozen transformer layer (requires_grad=False)
        self.transformer_block = nn.Linear(768, 768)
        self.transformer_block.requires_grad_(False)
        
        # Simulate LoRA-style trainable adapter (requires_grad=True)
        self.adapter = nn.Linear(768, 768)
        
    def forward(self, x):
        x = self.transformer_block(x)  # Frozen parameters
        x = self.adapter(x)            # Trainable parameters
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 16, 768, dtype=torch.float32)

