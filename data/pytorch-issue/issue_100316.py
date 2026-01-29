# torch.rand(B, seq_len, hidden_size, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(MyModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.inv_scale = nn.Parameter(torch.Tensor([1 / self.head_size ** 0.5]), requires_grad=False)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        batch_size, seq_len, hidden_size = query.size()
        query = query.view(batch_size, seq_len, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_size).permute(0, 2, 3, 1)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        attention_weights = torch.matmul(query, key).div(self.inv_scale).softmax(dim=-1)
        output = torch.matmul(attention_weights, value)
        return output

def my_model_function():
    return MyModel(hidden_size=16, num_heads=1)

def GetInput():
    return torch.randn(1, 4, 16)

