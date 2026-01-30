import torch

print(torch.cuda.device_count()) # --> 0
print(torch.version.cuda) # --> 10.0.130