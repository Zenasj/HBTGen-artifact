# torch.rand(1, 3, 224, 224, dtype=torch.float32)  # Each input tensor in the tuple has this shape
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, inputs):
        a, b = inputs  # Unpack input tuple of two tensors
        return torch.add(a, b)

def my_model_function():
    return MyModel()

def GetInput():
    # Return a tuple of two random tensors matching the expected input shape
    return (torch.rand(1, 3, 224, 224), torch.rand(1, 3, 224, 224))

