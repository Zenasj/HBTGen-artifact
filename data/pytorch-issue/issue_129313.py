# torch.randint(0, 4, (B,), dtype=torch.long)  # Input shape: batch of indices for Embedding
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=4, embedding_dim=1)
        # Initialize weights as in the original code to reproduce the issue
        with torch.no_grad():
            self.embedding.weight[0, 0] = 0.0
            self.embedding.weight[1, 0] = 1.0
            self.embedding.weight[2, 0] = 2.0
            self.embedding.weight[3, 0] = 3.0

    def forward(self, x):
        return self.embedding(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Generate random indices within the embedding's num_embeddings (0-3)
    batch_size = 2  # Example batch size, can be adjusted
    return torch.randint(0, 4, (batch_size,), dtype=torch.long)

