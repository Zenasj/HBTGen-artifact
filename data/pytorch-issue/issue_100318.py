import torch.nn as nn

py
import math
import torch

torch.manual_seed(420)

input_dim = 4
sequence_length = 3
batch_size = 2
num_heads = 2
hidden_dim = 1
dropout_p = 1

inputs = torch.randn(batch_size, sequence_length, input_dim)
attn_mask = torch.zeros(batch_size, num_heads, sequence_length, sequence_length).bool()

class Attention(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, num_heads, dropout_p):
        super().__init__()
        self.query = torch.nn.Linear(input_dim, hidden_dim * num_heads, bias=False)
        self.key = torch.nn.Linear(input_dim, hidden_dim * num_heads, bias=False)
        self.value = torch.nn.Linear(input_dim, hidden_dim * num_heads, bias=False)
        self.dropout = torch.nn.Dropout(dropout_p)
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

    def forward(self, inputs, attn_mask=None):
        q = self.query(inputs).view(inputs.size(0), self.num_heads, -1, self.hidden_dim)
        k = self.key(inputs).view(inputs.size(0), self.num_heads, -1, self.hidden_dim)
        v = self.value(inputs).view(inputs.size(0), self.num_heads, -1, self.hidden_dim)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.hidden_dim)
        scores = scores + attn_mask
        attn_weights = torch.softmax(scores, dim=-1)
        output = attn_weights @ v
        return output

func = Attention(input_dim, hidden_dim, num_heads, dropout_p).to('cpu')

res1 = func(inputs, attn_mask)
print(res1)
# tensor([[[[ 0.0261],
#           [ 0.0116],
#           [ 0.0387]],
# 
#          [[-0.0424],
#           [-0.0398],
#           [-0.0697]]],
# 
# 
#         [[[-0.1785],
#           [-0.1944],
#           [-0.1605]],
# 
#          [[-0.5129],
#           [-0.5133],
#           [-0.5141]]]], grad_fn=<UnsafeViewBackward0>)  

jit_func = torch.compile(func)
res2 = jit_func(inputs, attn_mask)
print(res2)
# tensor([[[[nan],
#           [nan],
#           [nan]],
# 
#          [[nan],
#           [nan],
#           [nan]]],
# 
# 
#         [[[nan],
#           [nan],
#           [nan]],
# 
#          [[nan],
#           [nan],
#           [nan]]]], grad_fn=<CompiledFunctionBackward>)