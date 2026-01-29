# torch.rand(4, 4, dtype=torch.float32)
import torch
import numpy as np

class MyModel(torch.nn.Linear):
    def __init__(self):
        # Compute input shape product explicitly as integer to avoid symbolic tracing issues
        in_features_val = np.array([4, 4, 1]).prod().item()
        super().__init__(in_features_val, in_features_val)  # Initialize Linear layer
        self.in_features_val = in_features_val  # Store as explicit integer attribute

    def forward(self, x):
        # Explicitly use stored integer to avoid FakeTensor wrapping during reshape
        return x.reshape((self.in_features_val,))

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand([4, 4], dtype=torch.float32)

