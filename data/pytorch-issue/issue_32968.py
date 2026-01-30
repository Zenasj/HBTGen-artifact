import torch

def triu_onnx(x):
     arange = torch.arange(len(x), device = x.device)
     mask = arange.unsqueeze(-1).expand(-1, len(x)) <= arange
     return x * mask

def triu_onnx(x, diagonal=0):
    l = x.shape[0]
    arange = torch.arange(l, device=x.device)
    mask = arange.expand(l, l)
    arange = arange.unsqueeze(-1)
    if diagonal:
        arange = arange + diagonal
    mask = mask >= arange
    return mask * x