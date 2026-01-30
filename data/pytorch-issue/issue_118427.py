import torch

def k(x):
    with torch.inference_mode():
        x = x + 1
        return x

torch.compile(k, backend="eager", fullgraph=True)(x)