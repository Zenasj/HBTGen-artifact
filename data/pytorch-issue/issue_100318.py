import math
import torch
from torch import nn

# torch.rand(B, sequence_length, input_dim, dtype=torch.float32) and a bool mask of shape (B, num_heads, sequence_length, sequence_length)
class MyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, dropout_p):
        super().__init__()
        self.query = nn.Linear(input_dim, hidden_dim * num_heads, bias=False)
        self.key = nn.Linear(input_dim, hidden_dim * num_heads, bias=False)
        self.value = nn.Linear(input_dim, hidden_dim * num_heads, bias=False)
        self.dropout = nn.Dropout(dropout_p)
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

    def forward(self, x):
        inputs, attn_mask = x  # Unpack the tuple input
        q = self.query(inputs).view(inputs.size(0), self.num_heads, -1, self.hidden_dim)
        k = self.key(inputs).view(inputs.size(0), self.num_heads, -1, self.hidden_dim)
        v = self.value(inputs).view(inputs.size(0), self.num_heads, -1, self.hidden_dim)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.hidden_dim)
        scores = scores + attn_mask
        attn_weights = torch.softmax(scores, dim=-1)
        output = attn_weights @ v
        return output

def my_model_function():
    # Return an instance of MyModel with the parameters from the issue's example
    return MyModel(input_dim=4, hidden_dim=1, num_heads=2, dropout_p=1)

def GetInput():
    # Return a tuple of inputs and attention mask matching the required shapes
    batch_size = 2
    sequence_length = 3
    input_dim = 4
    num_heads = 2
    inputs = torch.randn(batch_size, sequence_length, input_dim)
    attn_mask = torch.zeros(batch_size, num_heads, sequence_length, sequence_length).bool()
    return (inputs, attn_mask)

