# torch.rand(B, 1, dtype=torch.long) ← Add a comment line at the top with the inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_to_learn = nn.Embedding(10, 5)
        self.embedding_to_not_learn = nn.Embedding(10, 5).requires_grad_(False)

    def forward(self, x):
        learned_embeddings = self.embedding_to_learn(x)
        not_learned_embeddings = self.embedding_to_not_learn(x)
        return learned_embeddings, not_learned_embeddings

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    B = 4  # Batch size
    return torch.randint(0, 10, (B, 1), dtype=torch.long)

