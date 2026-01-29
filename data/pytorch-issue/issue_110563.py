# torch.randint(0, 15, (B, S), dtype=torch.long)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(15, 96)
        enc_layer = nn.TransformerEncoderLayer(
            96,
            nhead=12,
            dim_feedforward=96,
            dropout=0.2,
            batch_first=True
        )
        self.attn_layers = nn.TransformerEncoder(
            enc_layer,
            num_layers=10,
            enable_nested_tensor=True  # From original code
        )

    def forward(self, x):
        x = self.emb(x)
        return self.attn_layers(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Matches sample_input shape (1, 3) and dynamic export requirements
    B, S = 1, 3  # From original sample_input [[1,2,3]]
    return torch.randint(0, 15, (B, S), dtype=torch.long)

