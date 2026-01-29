# torch.rand(B, 20, 10, dtype=torch.float)  # Input shape: batch x sequence_length x features
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_layer = nn.Linear(10, 10)  # Frozen base layer
        self.lora_layer = nn.Linear(10, 1)   # Trainable LoRA-like layer
        # Freeze base parameters (mimics PEFT's behavior)
        for param in self.base_layer.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.base_layer(x)  # Process input through frozen base
        x = self.lora_layer(x)  # Apply trainable LoRA layer
        return x

def my_model_function():
    return MyModel()

def GetInput():
    B = 2  # Example batch size
    seq_len = 20  # Example sequence length
    return torch.rand(B, seq_len, 10, dtype=torch.float)

