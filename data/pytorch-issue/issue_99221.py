# torch.randint(0, 2, (1,), dtype=torch.int64)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(2, 3, dtype=torch.float32))  # Matches the example's weight shape [2,3]
        self.register_buffer('offsets', torch.as_tensor([0], dtype=torch.int64))  # Fixed offsets from example

    def forward(self, input):
        return F.embedding_bag(input, self.weight, self.offsets)

def my_model_function():
    return MyModel()

def GetInput():
    # Returns valid indices (0 or 1) to avoid the error shown in the issue
    return torch.randint(0, 2, (1,), dtype=torch.int64)

