# torch.randint(0, 32000, (1, 10), dtype=torch.long)  # Input shape (batch_size, sequence_length)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Simplified structure mimicking Flan-T5 components
        # Embedding layer (vocab size ~32k for T5, hidden_size=768)
        self.embeddings = nn.Embedding(32000, 768)
        # Dummy encoder layers (actual T5 has transformer blocks)
        self.encoder = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 768)
        )

    def forward(self, input_ids):
        # Simplified forward pass
        embeddings = self.embeddings(input_ids)
        return self.encoder(embeddings)

def my_model_function():
    # Returns a simple model instance with random weights
    return MyModel()

def GetInput():
    # Generates valid input_ids (long tensor) matching T5's expected format
    return torch.randint(0, 32000, (1, 10), dtype=torch.long)

