# torch.rand(B, S, dtype=torch.long)  # Input shape (batch, sequence_length) for embedding layer
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Simulate LLaMA-like architecture with frozen embedding layer and trainable linear layer
        self.embedding = nn.Embedding(num_embeddings=10000, embedding_dim=5120)  # Typical LLaMA vocab size and hidden dim
        self.embedding.weight.requires_grad_(False)  # Frozen layer to trigger requires_grad inconsistency
        self.linear = nn.Linear(5120, 5120)  # Example transformer block component

    def forward(self, x):
        x = self.embedding(x)
        return self.linear(x)

def my_model_function():
    # Returns model instance with mixed requires_grad parameters (required to reproduce state_dict error)
    return MyModel()

def GetInput():
    # Generate random input tensor matching embedding layer expectations
    batch_size = 2
    seq_length = 1024  # Typical LLaMA sequence length
    return torch.randint(low=0, high=10000, size=(batch_size, seq_length), dtype=torch.long)

