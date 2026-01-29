# torch.rand(S, B, E, dtype=torch.float32)  # Example input shape: (5, 2, 8)
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class MyModel(nn.Module):
    def __init__(self, d_model=8, nhead=2, num_layers=2):
        super(MyModel, self).__init__()
        # Original problematic encoder (clones layers)
        orig_layer = TransformerEncoderLayer(d_model, nhead)
        self.orig_encoder = TransformerEncoder(orig_layer, num_layers)
        
        # Fixed encoder (each layer initialized uniquely)
        self.fixed_encoder = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead) for _ in range(num_layers)
        ])
        
    def forward(self, x):
        # Run original encoder (cloned layers)
        orig_out = self.orig_encoder(x)
        
        # Run fixed encoder (unique layers)
        fixed_out = x
        for layer in self.fixed_encoder:
            fixed_out = layer(fixed_out)
            
        # Return 1.0 if outputs differ, 0.0 otherwise
        return torch.tensor([1.0 if not torch.allclose(orig_out, fixed_out, atol=1e-6) else 0.0], dtype=torch.float32)

def my_model_function():
    return MyModel()

def GetInput():
    # Generate random input matching (sequence_length, batch_size, embedding_dim) convention
    return torch.rand(5, 2, 8, dtype=torch.float32)

