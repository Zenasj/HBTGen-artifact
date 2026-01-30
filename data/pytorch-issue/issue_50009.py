import torch

model.eval()

with torch.no_grad():
    model.decoder.store_inverse()