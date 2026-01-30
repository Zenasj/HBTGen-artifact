import torch

device = 'mps'

mag = torch.randn(512,512).to(device)
phase = torch.randn(512,512).to(device)

out = mag*(torch.cos(mag)+1j* torch.sin(phase))