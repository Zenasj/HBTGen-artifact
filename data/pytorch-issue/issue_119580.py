# torch.rand(B, S, E, dtype=torch.float32)  # Example input shape: (1, 4, 8)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        B, S, E = x.size()
        query = x
        key = x
        value = x
        mask = torch.tril(torch.ones(S, S, device=x.device)).bool()
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=mask, dropout_p=0.0
        )

def my_model_function():
    return MyModel()

def GetInput():
    B, S, E = 1, 4, 8  # Example dimensions
    return torch.rand(B, S, E, dtype=torch.float32)

