# torch.rand(1, 512, 1024, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1024, 1024, bias=False)
        self.transformer = nn.TransformerEncoderLayer(
            d_model=1024, nhead=16, dim_feedforward=4096, dropout=0.1
        )
    
    def forward(self, x):
        # Process Linear layer input directly
        out_linear = self.linear(x)
        
        # Process Transformer input (requires (seq_len, batch, features))
        x_transformer = x.permute(1, 0, 2)  # Convert (B, S, E) → (S, B, E)
        out_transformer = self.transformer(x_transformer)
        
        return out_linear, out_transformer

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 512, 1024, dtype=torch.float32)

