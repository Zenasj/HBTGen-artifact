# torch.randint(0, 50257, (1, 14), dtype=torch.int64)  # Input shape inferred from error logs (B=1, S=14)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # BLOOM-560M has vocab_size=250887 and hidden_size=1024, but simplified here for minimal repro
        self.word_embeddings = nn.Embedding(50257, 512)  # Approximate embedding layer
        self.transformer_block = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )  # Dummy transformer block to mimic model structure

    def forward(self, input_ids):
        embeddings = self.word_embeddings(input_ids)
        return self.transformer_block(embeddings)

def my_model_function():
    # Returns a minimal model with embedding layer and dummy transformer blocks
    return MyModel()

def GetInput():
    # Generates input_ids tensor matching BLOOM's expected format
    return torch.randint(0, 50257, (1, 14), dtype=torch.int64)

