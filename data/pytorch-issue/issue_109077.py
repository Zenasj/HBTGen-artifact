# torch.randint(0, 100, (B, S), dtype=torch.long)  # Input is token indices for embedding
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Base LLM components (simplified example)
        self.embedding = nn.Embedding(100, 256)       # Original model's embedding layer
        self.linear = nn.Linear(256, 512)             # Example transformer layer
        # Prompt tuning addition (e.g., extra embedding for prompt tokens)
        self.prompt_embedding = nn.Embedding(32, 256) # Newly added prompt parameters

    def forward(self, x):
        # Forward pass using base model components (prompt embedding unused in this example)
        return self.linear(self.embedding(x))

def my_model_function():
    # Returns model with prompt tuning parameters (prompt_embedding)
    return MyModel()

def GetInput():
    # Generates random token indices for input
    B = 2    # Batch size
    S = 10   # Sequence length
    return torch.randint(0, 100, (B, S), dtype=torch.long)

