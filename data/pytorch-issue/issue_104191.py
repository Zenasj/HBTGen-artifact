# torch.randint(0, 50257, (B, S), dtype=torch.long)  # Input shape: (batch, sequence_length)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # GPT-2 style embedding layer with BFloat16 dtype (problematic on MPS)
        self.embedding = nn.Embedding(
            num_embeddings=50257,  # GPT-2 vocab size
            embedding_dim=768,     # Standard embedding dimension
            dtype=torch.bfloat16   # Problematic dtype for MPS
        )

    def forward(self, input_ids):
        return self.embedding(input_ids)

def my_model_function():
    model = MyModel()
    # Attempt to move to MPS if available (will trigger error on unsupported systems)
    if torch.backends.mps.is_available():
        model.to("mps")
    return model

def GetInput():
    # Generate random input tensor matching GPT-2's expected input
    batch_size = 1
    sequence_length = 5  # Minimal test sequence length
    return torch.randint(
        0, 50257,  # Valid token IDs for GPT-2
        (batch_size, sequence_length),
        dtype=torch.long  # Required for embedding layer input
    )

