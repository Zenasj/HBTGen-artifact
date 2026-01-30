import torch

device = torch.device('cuda') # or 'cpu'

# Note that we should have passed device.type here, so the fact that this errors isn't itself a bug.
with torch.autocast(device_type=device):
    pass