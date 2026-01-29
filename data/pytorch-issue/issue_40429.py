# torch.rand(B, C, dtype=torch.float)
import torch
import torch.nn as nn

class MyBatchNorm1d(nn.BatchNorm1d):
    def __init__(self, num_features, affine=True, use_scale=True, **kwargs):
        super().__init__(num_features, affine=affine, **kwargs)
        if affine and not use_scale:
            self.weight.data.fill_(1.0)
            self.weight.requires_grad_(False)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = MyBatchNorm1d(3, affine=True, use_scale=False)  # Matches the example's use_scale=False

    def forward(self, x):
        return self.bn(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Return a random input tensor matching the BatchNorm1d's expected input shape (Batch, Channels)
    return torch.rand(2, 3, dtype=torch.float)

