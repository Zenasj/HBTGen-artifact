import torch

def _fill_aten(a, value):
    t = a * False
    with torch.no_grad():
        t.fill_(value)
    return t