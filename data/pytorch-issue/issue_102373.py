import torch

if device.type == "cuda" and torch.cuda.get_device_capability(device) < (6, 0):
    torch.cuda.set_device(device)