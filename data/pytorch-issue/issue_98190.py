# torch.rand(B, 32, dtype=torch.long)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(100, 32)  # Inferred vocabulary size and embedding dim
        self.fc = nn.Linear(32, 64)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.embedding(x)  # Process long tensor input
        x = x.mean(dim=1)      # Aggregate over sequence dimension (B,32)
        x = self.fc(x)         # (B,64)
        # Unflatten operation causing ONNX export issue
        x = x.unflatten(1, (8, 8))  # Split into (B,8,8)
        # Attention layer using scaled_dot_product_attention
        attn_output = F.scaled_dot_product_attention(
            x, x, x, 
            dropout_p=0.1,
            is_causal=False
        )
        x = self.dropout(attn_output)
        return x.view(x.size(0), -1)  # Flatten output

def my_model_function():
    return MyModel()

def GetInput():
    # Matches the (B,32) input shape from issue's dummy_input
    B = 1  # Batch size placeholder
    return torch.randint(0, 100, (B, 32), dtype=torch.long)

