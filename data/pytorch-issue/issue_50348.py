# torch.rand(B, 1, H, dtype=torch.float32)  # B: Batch size, H: Sequence length

import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, ntaps):
        super(MyModel, self).__init__()
        self.loc = Localiser(ntaps)
        self.ntaps = ntaps

    def forward(self, x):
        filter = self.loc(x)
        filter = filter.view(-1, self.ntaps)
        x = adaptive_filter(x, filter, self.ntaps)
        return x

class Localiser(nn.Module):
    def __init__(self, nfeats):
        super(Localiser, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 128, 3, padding=1),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(128, nfeats)

    def forward(self, x):
        x = self.net(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

@torch.jit.script
def adaptive_filter(x, w, padding: int):
    B = x.shape[0]
    x = x.permute(1, 0, 2)
    w = w.unsqueeze(1)
    x = F.conv1d(x, w, padding=padding, groups=B)
    x = x.permute(1, 0, 2)
    return x

def my_model_function():
    return MyModel(13)

def GetInput():
    B = 4  # Example batch size
    H = 1024  # Example sequence length
    return torch.randn(B, 1, H, dtype=torch.float32)

