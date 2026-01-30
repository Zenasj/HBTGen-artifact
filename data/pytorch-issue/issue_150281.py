import torch

# If comment out `torch.utils.rename_privateuse1_backend("test")`, works fine.
torch.utils.rename_privateuse1_backend("test")

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU], 
) as p:
    pass