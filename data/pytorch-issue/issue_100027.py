# torch.rand(B, S, H, dtype=torch.float32)
import math
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, num_heads, hidden_dim):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(hidden_dim, hidden_dim * num_heads)
        self.key = nn.Linear(hidden_dim, hidden_dim * num_heads)
        self.value = nn.Linear(hidden_dim, hidden_dim * num_heads)
        self.scale = math.sqrt(hidden_dim // num_heads)  # Fixed to float instead of Parameter

    def forward(self, x):
        b, s, h = x.size()
        query = self.query(x).view(b, s, self.num_heads, self.hidden_dim // self.num_heads).transpose(1, 2)
        key = self.key(x).view(b, s, self.num_heads, self.hidden_dim // self.num_heads).transpose(1, 2)
        value = self.value(x).view(b, s, self.num_heads, self.hidden_dim // self.num_heads).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(2, 3)) / self.scale
        attn_probs = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_probs, value)
        return attn_output

def my_model_function():
    # Initialize with parameters from the original issue (num_heads=1, hidden_dim=512)
    return MyModel(num_heads=1, hidden_dim=512)

def GetInput():
    # Input shape matching the original test case (batch=4, sequence=16, hidden=512)
    return torch.randn(4, 16, 512)

