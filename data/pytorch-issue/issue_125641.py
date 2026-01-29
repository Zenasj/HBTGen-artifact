# torch.rand(B, C, H, W, dtype=...)  # Inferred input shape: (batch_size, sequence_length)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.vocab_embed = nn.Embedding(vocab_size, hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor):
        out = self.vocab_embed(x)
        out = self.head(out)
        return out

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel(vocab_size=8000, hidden_size=2048)

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    batch_size = 16
    sequence_length = 1024
    vocab_size = 8000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input = torch.randint(0, vocab_size - 1, (batch_size, sequence_length), device=device)
    return input

