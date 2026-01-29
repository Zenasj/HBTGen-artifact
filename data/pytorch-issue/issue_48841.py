# torch.rand(B, C, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, num_samples=1, replacement=False):
        super().__init__()
        self.num_samples = num_samples
        self.replacement = replacement

    def forward(self, probabilities):
        return torch.multinomial(probabilities, self.num_samples, self.replacement)

def my_model_function():
    return MyModel()  # Default configuration from the original bug report (num_samples=1, replacement=False)

def GetInput():
    B = 100000  # Batch size from the first example
    C = 2       # Number of classes (columns)
    probabilities = torch.zeros(B, C, dtype=torch.float32)
    probabilities[:, 1] = 1.0  # Set second column to 1.0 (valid), first column to 0.0 (invalid)
    return probabilities

