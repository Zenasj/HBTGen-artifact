# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Dynamo guard condition simplified to check batch size <3
        if x.size(0) < 3:
            return x * 2  # Example operation path for small batch
        else:
            return x * 1  # Alternate path for larger batch

def my_model_function():
    return MyModel()

def GetInput():
    # Matches input shape with batch_size=2 (to trigger the simplified guard)
    return torch.rand(2, 2, 1, 1, dtype=torch.float32)

