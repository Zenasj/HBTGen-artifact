import torch.nn as nn

py
import torch

torch.manual_seed(420)

class Attention(torch.nn.Module):

    def __init__(self, hidden_size, num_heads):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.query = torch.nn.Linear(hidden_size, hidden_size)
        self.key = torch.nn.Linear(hidden_size, hidden_size)
        self.value = torch.nn.Linear(hidden_size, hidden_size)
        self.inv_scale = torch.nn.Parameter(torch.Tensor([1 / self.head_size ** 0.5]), requires_grad=False)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        (batch_size, seq_len, hidden_size) = query.size()
        query = query.view(batch_size, seq_len, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_size).permute(0, 2, 3, 1)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        attention_weights = torch.matmul(query, key).div(self.inv_scale).softmax(dim=-1)
        # print(attention_weights.shape)
        # print(value.shape)
        output = torch.matmul(attention_weights, value)
        return output

hidden_size = 16
num_heads = 1
seq_len = 4
batch_size = 1
x = torch.randn(batch_size, seq_len, hidden_size)

func = Attention(hidden_size, num_heads).to('cpu')

with torch.no_grad():
    res1 = func(x)
    print(res1)

    jit_func = torch.compile(func)
    res2 = jit_func(x)
    print(res2)
    # RuntimeError: Expected size for first two dimensions of batch2 tensor to be: [4, 1] but got: [4, 4].