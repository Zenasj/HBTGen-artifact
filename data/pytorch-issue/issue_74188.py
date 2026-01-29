# torch.rand(B, C, H, W, dtype=...)  # Inferred input shape: (N_EXAMPLES_PER_BATCH, N_FEATURES, 1)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.flatten = nn.Flatten()
        self.nn = nn.Linear(in_features=1, out_features=1)

    def forward(self, x):
        x = self.flatten(x)
        return self.nn(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    N_EXAMPLES_PER_BATCH = 32
    N_FEATURES = 1
    return torch.rand(N_EXAMPLES_PER_BATCH, N_FEATURES, 1)

