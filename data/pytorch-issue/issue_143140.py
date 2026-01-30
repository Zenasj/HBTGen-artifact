import torch

with torch.autocast('mps', enabled=False):
       result = torch.einsum("bixyz,ioxyz->boxyz", input_tensor, weights)