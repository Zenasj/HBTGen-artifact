# torch.randint(0, 50257, (1, 20), dtype=torch.long)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # GPT2-large parameters: vocab_size=50257, hidden_size=1024
        self.embedding = nn.Embedding(50257, 1024)
        self.linear = nn.Linear(1024, 1024)  # Simplified layer for BLAS usage
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.linear(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    # Random input tokens matching GPT2's expected input shape
    return torch.randint(0, 50257, (1, 20), dtype=torch.long)

