import torch
size = 4096
nan_tensor = torch.nan * torch.ones(size, size)
nan_tensor = nan_tensor.to("cuda")
# The following doesn't throw an error, but deadlocks the GPU indefinitely:
torch.linalg.inv(nan_tensor)