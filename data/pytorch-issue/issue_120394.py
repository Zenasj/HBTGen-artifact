# torch.randint(low=0, high=128, size=(1, 10), dtype=torch.long)  # Add a comment line at the top with the inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.embedding_bag = nn.EmbeddingBag(num_embeddings=128, embedding_dim=32)

    def forward(self, inputs):
        return self.embedding_bag(inputs)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randint(low=0, high=128, size=(1, 10), dtype=torch.long)

