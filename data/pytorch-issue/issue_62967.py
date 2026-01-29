# torch.rand(B, C, H, W, dtype=torch.float32)  # e.g., B=2, C=2, H=10, W=1

import torch
import torch.nn as nn

class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(self.dims)

class MyModel(nn.Module):
    def __init__(self, num_features=10):
        super().__init__()
        # Original BatchNorm2d path (permute + BN + permute back)
        self.bn_path = nn.Sequential(
            Permute(dims=(0, 2, 1, 3)),
            nn.BatchNorm2d(num_features=num_features),
            Permute(dims=(0, 2, 1, 3)),
        )
        
        # SyncBatchNorm path created by converting a copy
        sync_base = nn.Sequential(
            Permute(dims=(0, 2, 1, 3)),
            nn.BatchNorm2d(num_features=num_features),
            Permute(dims=(0, 2, 1, 3)),
        )
        self.sync_path = torch.nn.SyncBatchNorm.convert_sync_batchnorm(sync_base)

    def forward(self, x):
        bn_out = self.bn_path(x)
        sync_out = self.sync_path(x)
        return (bn_out, sync_out)  # Return both outputs for comparison

def my_model_function():
    return MyModel(num_features=10)

def GetInput():
    return torch.randn(2, 2, 10, 1, dtype=torch.float32)

