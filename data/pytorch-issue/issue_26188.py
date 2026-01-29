# torch.randint(0, 10, (B, 3), dtype=torch.long)  # Inferred input shape (B=batch_size, 3=num_embeddings)
import torch
import numpy as np
from torch import nn

class MyModel(nn.Module):
    def __init__(self, *embeddings):
        super(MyModel, self).__init__()
        self.embeddings = nn.ModuleList()
        for embedding in embeddings:
            embedding = torch.as_tensor(embedding)
            embedding = nn.Embedding.from_pretrained(embedding, freeze=True)
            self.embeddings.append(embedding)
    
    def forward(self, x):
        # x: (batch_size, num_embeddings) indices tensor
        # Output: sum of embedded vectors across all embeddings
        embedded = [emb(x[:, i]) for i, emb in enumerate(self.embeddings)]
        return torch.stack(embedded, dim=1).sum(dim=1)

def my_model_function():
    # Create small embeddings for demonstration (original size: (5000000, 1000))
    # Using 10x10 tensors to avoid excessive memory usage in code
    embeddings = [np.zeros((10, 10), dtype=np.float32) for _ in range(3)]
    return MyModel(*embeddings)

def GetInput():
    B = 2  # Example batch size
    return torch.randint(0, 10, (B, 3), dtype=torch.long)

