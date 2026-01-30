import torch
import torch.nn as nn

# some forward code

with torch.backends.cudnn.flags(enabled=False):
    out = problematic_module(inputs)

# more forward code