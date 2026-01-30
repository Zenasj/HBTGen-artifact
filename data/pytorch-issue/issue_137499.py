import torch.nn as nn

# repro.py
import torch
torch.set_default_device("cuda")
torch.set_grad_enabled(False)

batch_size = 32
seq_length = 50
hidden_size = 768

inp = torch.randn(batch_size, seq_length, hidden_size)
weight = torch.randn(hidden_size, hidden_size)

layer_norm = torch.nn.LayerNorm(hidden_size)

@torch.compile()
def foo(inp, weight):
    matmul_output = inp @ weight
    final_output = layer_norm(matmul_output)
    return final_output

print(foo(inp, weight))