# torch.randint(0, 2, (B,), dtype=torch.long)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.register_buffer('weight', weight)  # Stores LongTensor as non-trainable buffer
    
    def forward(self, input):
        return torch.nn.functional.embedding(input, self.weight)

def my_model_function():
    # Initialize with LongTensor weight to demonstrate scenario
    weight = torch.LongTensor([[1, 2, 3], [4, 5, 6]])  # Matches example's 2x3 structure
    return MyModel(weight)

def GetInput():
    # Returns random indices within valid range (0-1 for 2 embeddings)
    return torch.randint(0, 2, (1,), dtype=torch.long)

