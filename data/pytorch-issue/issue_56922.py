import torch
import torch.nn as nn

with torch.backends.mkldnn.flags(enabled=False):
    x = conv(sample)