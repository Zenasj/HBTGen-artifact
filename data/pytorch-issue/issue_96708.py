import torch

def test_fn(x,b,c):
    a = torch.ops.quantized_decomposed.quantize_per_tensor.default(x, 0.1, 0, b, c, torch.int8)
    return a