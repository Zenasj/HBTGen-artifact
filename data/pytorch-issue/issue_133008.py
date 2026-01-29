# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Parameters from __add__.pt (assuming parameter:0 and parameter:1 are tensors)
        self.add_param0 = nn.Parameter(torch.randn(1, 3, 224, 224))  # Example shape
        self.add_param1 = nn.Parameter(torch.randn(1, 3, 224, 224))
        # Parameters from layer_norm.pt (normalized_shape is a tuple, weight/bias are tensors)
        self.norm_shape = (3, 224, 224)  # Example normalized_shape
        self.weight = nn.Parameter(torch.randn(3, 224, 224))  # layer_norm weight
        self.bias = nn.Parameter(torch.randn(3, 224, 224))    # layer_norm bias
        self.eps = 1e-5  # Example epsilon value
        # Parameters from __add__2.pt (parameter:1 is a tensor)
        self.add2_param = nn.Parameter(torch.randn(1, 3, 224, 224))

    def forward(self, x):
        # First __add__: parameter0 + parameter1 (from __add__.pt)
        intermediate = self.add_param0 + self.add_param1
        # LayerNorm: apply to the intermediate result
        normed = F.layer_norm(intermediate, self.norm_shape, self.weight, self.bias, self.eps)
        # Second __add__: add parameter from __add__2.pt
        output = normed + self.add2_param
        return output

def my_model_function():
    return MyModel()

def GetInput():
    # Return a random input tensor matching the expected shape (B, C, H, W)
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

