# torch.rand(2, 129, 100, 96, dtype=torch.float32)  # BxFxTxH shape
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        for _ in range(2):
            layer = nn.ModuleList([
                nn.LayerNorm(96),
                nn.MultiheadAttention(embed_dim=96, num_heads=4, batch_first=True)
            ])
            layers.append(layer)
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        B, F, T, H = x.shape
        for m in self.layers:
            x = x.reshape(B * F, T, H)
            x = m[0](x)  # LayerNorm
            x, _ = m[1](x, x, x)  # MultiheadAttention (ignores attention weights)
            x = x.reshape(B, F, T, H)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 129, 100, 96, dtype=torch.float32)

