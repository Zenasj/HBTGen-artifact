import torch.nn as nn

py
import math
import torch

torch.manual_seed(420)


class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_mask: torch.Tensor):
        attn_weight = torch.softmax(query @ key.transpose(-2, -1) / math.sqrt(query.size(-1)) + attn_mask, dim=-1)
        out = attn_weight @ value
        return out

hidden_size = 8
sequence_length = 10
num_heads = 2
batch_size = 4

query = torch.randn(batch_size, num_heads, sequence_length, hidden_size)
key = torch.randn(batch_size, num_heads, sequence_length, hidden_size)
attn_mask = torch.zeros(batch_size, num_heads, sequence_length, sequence_length)
positional_encoding = torch.arange(0, sequence_length).unsqueeze(0).unsqueeze(0)

func = Model().to('cpu')
test_inputs = [query, key, attn_mask, positional_encoding]

with torch.no_grad():
    res1 = func(*test_inputs)
    print(res1)

    jit_func = torch.compile(func)
    res2 = jit_func(*test_inputs)
    print(res2)
    # torch._dynamo.exc.BackendCompilerFailed: backend='debug_wrapper' raised:
    # RuntimeError: Expected attn_mask dtype to be bool or to match query dtype, but got attn_mask.dtype: long int and  query.dtype: float instead.