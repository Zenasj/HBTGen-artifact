# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        t = x  # First input to the custom op
        tensors = [x, x]  # List of tensors as second input (simulated input structure)
        # Simulate FB's custom op returning (Tensor, List[Tensor])
        out1, out_list = self._simulate_custom_op(t, tensors)
        return out1 + out_list[0] + out_list[1]  # Example aggregation of outputs
    
    def _simulate_custom_op(self, t, tensors):
        # Dummy implementation matching the described return signature
        # Returns (Tensor, List[Tensor]) for demonstration purposes
        return t, tensors  # tensors is a list of two tensors as shown in the issue example

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)  # Standard 4D input tensor

