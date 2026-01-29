# torch.rand(B, S, dtype=torch.long)  # Input shape: (batch_size, sequence_length)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(10000, 768)  # Example GPT2-like embedding layer
        self.fc = nn.Linear(768, 768)  # Simplified transformer block component

    def forward(self, x):
        x = self.embedding(x)
        x = self.fc(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    B = 2  # Batch size
    S = 10  # Sequence length
    return torch.randint(0, 10000, (B, S), dtype=torch.long)

