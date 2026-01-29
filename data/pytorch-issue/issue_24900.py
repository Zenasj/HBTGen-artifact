# torch.rand(B, C, H, W, dtype=...)  # In this case, the input is a 2D tensor of shape (N, 4)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, X):
        # Transpose the input to avoid memory allocation issues with large N
        X = X.T
        U, S, V = torch.svd(X, some=True)
        return U, S, V

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate a random tensor input that matches the input expected by MyModel
    N = 150000
    X = torch.randn((N, 4))
    return X

