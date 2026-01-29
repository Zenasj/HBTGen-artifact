# Input: a tuple of (indices of shape (N,), offsets of shape (B+1,)), both LongTensors
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.embedding_bag = nn.EmbeddingBag(5, 3)  # Matches EmbeddingBag(5,3) from the issue examples

    def forward(self, x):
        indices, offsets = x
        return self.embedding_bag(indices, offsets)

def my_model_function():
    # Returns an instance with default-initialized EmbeddingBag
    return MyModel()

def GetInput():
    # Reproduces input from first issue example (empty sequence in middle of batch)
    indices = torch.LongTensor([0, 1, 2, 3, 4])
    offsets = torch.LongTensor([0, 1, 1, 3, 3, 4])  # Creates empty sequence between offsets[1] and offsets[2]
    return (indices, offsets)

