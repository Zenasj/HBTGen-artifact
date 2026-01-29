import torch
from torch import nn

# torch.randint(0, 151936, (B, S), dtype=torch.long)
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        vocab_size = 151936  # Inferred from shape [151936, 896] in issue comments
        embed_dim = 896
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        # Explicitly tie word embeddings and lm_head weights
        self.lm_head.weight = self.embeddings.weight  # Critical for synchronization

    def forward(self, input_ids):
        # Simplified forward pass to match minimal model structure
        embeddings = self.embeddings(input_ids)
        return self.lm_head(embeddings)

def my_model_function():
    # Returns an instance of MyModel with tied weights
    return MyModel()

def GetInput():
    # Generates random input tensor matching the expected input shape
    B, S = 2, 10  # Example batch size and sequence length
    return torch.randint(0, 151936, (B, S), dtype=torch.long)

