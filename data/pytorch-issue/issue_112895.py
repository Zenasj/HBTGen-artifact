import torch

# I am using this gpustat and ipdb to report used GPU memory
# GPU memory : 449 mb
a = torch.empty(32000, 768).to("cuda") # GPU memory : 1863 mb
torch.median(a) # GPU memory : 9453 mb (and remains on this level)

3
import torch

with torch.no_grad():
    # Note the 320_000 instead of 32_000
    a = torch.ones(320000, 768, requires_grad=False).cuda()
    b = torch.median(a)
# Endless loop, so we can observe memory usage in nvidia-smi
while True:
    pass