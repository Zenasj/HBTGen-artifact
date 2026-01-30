import torch

py
emb = torch.rand(1, 6 ,64)
with torch.autocast(device_type="maia"):
    cos = emb.cos()