import torch
import torch.nn as nn

# torch.rand(B, 128, 512, dtype=torch.float)  # Inferred input shape for a sequence-based model
class CodeGenBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(512, 512)
        self.norm = nn.LayerNorm(512)
    
    def forward(self, x):
        return self.norm(self.linear(x))

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # FSDP configuration in the issue specifies wrapping CodeGenBlock layers
        self.transformer_layers = nn.ModuleList([CodeGenBlock() for _ in range(2)])
        self.output_layer = nn.Linear(512, 10)  # Example output layer
    
    def forward(self, x):
        for layer in self.transformer_layers:
            x = layer(x)
        return self.output_layer(x)

def my_model_function():
    # Initialize model with default settings (no custom weights needed)
    return MyModel()

def GetInput():
    # Generate random input tensor matching the model's expected input shape
    B = 2  # Batch size (matches num_processes: 2 in FSDP config)
    return torch.rand(B, 128, 512, dtype=torch.float)  # (batch, sequence_length, embedding_dim)

