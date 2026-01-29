# torch.rand(B, C, H, W, dtype=...)  # This line is not applicable for the given model. The input shape is (B, seq_len) where B is batch size and seq_len is the sequence length.
import torch.nn as nn
import torch

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(50, 64)
    
    def forward(self, x):
        inp = x.new_zeros(x.shape, dtype=x.dtype)  # Explicitly set the dtype to match the input's dtype
        return self.emb(inp)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # The input is a tensor of shape (B, seq_len) with dtype=torch.int64
    B, seq_len = 2, 3
    return torch.randint(0, 50, (B, seq_len), dtype=torch.int64)

