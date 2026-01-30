import torch
with torch.autocast("mps", dtype=torch.bfloat16):
    x = torch.tensor(1)

# /Users/hvaara/dev/pytorch/pytorch/torch/amp/autocast_mode.py:332: UserWarning: In MPS autocast, but the target dtype is not supported. Disabling autocast.
# MPS Autocast only supports dtype of torch.bfloat16 currently.
#   warnings.warn(error_message)