# torch.rand(1, 3, 128, 128, dtype=torch.float32)  # Inferred input shape
import torch
import numpy as np
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return x  # Identity model to trigger CUDA memory check

def my_model_function():
    return MyModel()

def GetInput():
    # Create non-contiguous tensor that triggers memory overlap error on CUDA
    numpy_matrix = np.empty((128, 128, 3))
    numpy_matrix = numpy_matrix.transpose((2, 0, 1))[np.newaxis]  # Shape becomes (1, 3, 128, 128)
    return torch.from_numpy(numpy_matrix).float()  # Returns non-contiguous tensor on CPU

