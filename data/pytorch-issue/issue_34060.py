# torch.rand(3, 2, 128, dtype=torch.float32)
import math
import torch
from torch import nn
from torch.nn.modules.activation import MultiheadAttention

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.att = MultiheadAttention(embed_dim=128, num_heads=8, dropout=0.)

    def forward(self, x):
        # Compute attention for full batch and first element subset
        output_full, _ = self.att(x, x, x)
        data_0 = x[:, 0:1]  # Extract first batch element as new batch of size 1
        output_single, _ = self.att(data_0, data_0, data_0)
        # Return difference between corresponding outputs
        return output_full[:, 0] - output_single[:, 0]

def my_model_function():
    return MyModel()

def GetInput():
    torch.manual_seed(42)
    return torch.randn(3, 2, 128) * math.sqrt(128)

