# torch.rand(B, S, dtype=torch.long)  # B=batch_size, S=sequence_length
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, vocab_size=30000, hidden_size=4096):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids):
        # Forward pass mimicking a causal LM's token prediction
        embeddings = self.embedding(input_ids)
        return self.linear(embeddings)

def my_model_function():
    # Initialize with bfloat16 as per the original model's dtype
    model = MyModel()
    model.to(dtype=torch.bfloat16)
    return model

def GetInput():
    # Generate random input_ids with typical Llama-2 vocab size and moderate sequence length
    B = 1  # Batch size from original script
    S = 20  # Example sequence length (inferred from context)
    return torch.randint(0, 30000, (B, S), dtype=torch.long)

