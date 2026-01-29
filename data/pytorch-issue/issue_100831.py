# torch.rand(1, 10, dtype=torch.int64) ← Add a comment line at the top with the inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, input_size, num_embeddings):
        super(MyModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=input_size)

    def forward(self, input_ids):
        embeddings = self.embedding(input_ids)
        return embeddings

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    input_size = 1
    num_embeddings = 100  # Set to a valid number to avoid IndexError
    return MyModel(input_size, num_embeddings)

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randint(0, 100, (1, 10), dtype=torch.int64)

