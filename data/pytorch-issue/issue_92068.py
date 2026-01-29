# torch.rand(1, dtype=torch.float32)  # Inferred minimal input shape
import torch
import torch.distributed as dist
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # The problematic call that triggers Dynamo tracing failure
        dist.is_available()
        return x  # Return input tensor to satisfy model requirements

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, dtype=torch.float32)  # Scalar tensor input

