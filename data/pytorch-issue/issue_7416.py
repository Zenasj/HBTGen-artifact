# torch.randint(0, 10000, (32,), dtype=torch.long)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Fixed parameters for reproducibility
        self.vocab_size = 10000
        self.active_size = 100
        self.embedding_dim = 10

        # Seed ensures deterministic active_vocab generation
        torch.manual_seed(42)
        active_vocab = torch.randperm(self.vocab_size)[:self.active_size]
        index = torch.arange(self.active_size, dtype=torch.long)
        
        # Create lookup table mapping active vocab indices to embedding positions
        lookup = torch.full((self.vocab_size,), self.active_size, dtype=torch.long)
        lookup[active_vocab] = index
        self.register_buffer('lookup', lookup)
        
        # Embedding layer for active vocabulary
        self.embedding = nn.Embedding(self.active_size, self.embedding_dim)

    def forward(self, input_indices):
        # Map input indices to active embedding indices
        active_indices = self.lookup[input_indices]
        return self.embedding(active_indices)

def my_model_function():
    return MyModel()

def GetInput():
    # Generate input using same seed to ensure indices are in active_vocab
    torch.manual_seed(42)
    active_vocab = torch.randperm(10000)[:100]
    batch_size = 32
    indices = active_vocab[torch.randint(0, 100, (batch_size,))]
    return indices.long()

