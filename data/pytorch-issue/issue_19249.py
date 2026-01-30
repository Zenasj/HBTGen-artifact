import torch.nn as nn

batch_norm = nn.BatchNorm2d
if dont_use_batch_norm:
    batch_norm = Identity

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        return x