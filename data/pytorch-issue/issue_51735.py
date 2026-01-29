# torch.rand(B, C, H, W, dtype=torch.float32)  # Input shape can be arbitrary (only device/dtype used)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, image):
        # Reproduces the issue with in-place normal_ mutation
        return image.new_empty(2, 1, 3, 4).normal_()

def my_model_function():
    # Returns the model instance causing the RemoveTensorMutation error
    return MyModel()

def GetInput():
    # Returns a dummy input tensor (shape not critical as it's not used in output dimensions)
    return torch.rand(2, 3, 4, 5)  # Example input with arbitrary shape

