# torch.rand(1, 1, 2, 2, dtype=torch.float32)  # Inferred input shape from test case
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        # Move input to CUDA and perform a simple operation to trigger CUDA activity
        return x.to('cuda') * 2.0

def my_model_function():
    return MyModel()

def GetInput():
    # Return a random CPU tensor (to be moved to CUDA in the model)
    return torch.rand(1, 1, 2, 2)  # Matches inferred input shape

