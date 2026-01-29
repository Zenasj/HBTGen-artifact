# torch.rand(B, C, H, W, dtype=...)  # Input shape: (512, 3, 144, 144), dtype=torch.float64
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # No specific parameters or layers needed for this model
        pass

    def forward(self, x):
        # Convert the input tensor to a sparse tensor
        x_sparse = x.to_sparse()
        # Compute the sum of the sparse tensor
        return torch.sparse.sum(x_sparse)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate a random tensor with the specified shape and dtype
    return torch.rand([512, 3, 144, 144], dtype=torch.float64)

