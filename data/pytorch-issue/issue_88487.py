# torch.randint(0, 1000, (B, S), dtype=torch.long)  # Example input: batch_size=1, seq_len=128
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(1000, 128)  # Example embedding layer

    def forward(self, x):
        emb = self.embedding(x)
        # Trigger the problematic 'unique_consecutive' op
        _, inv_indices = torch.unique_consecutive(emb, return_inverse=True)
        return inv_indices

def my_model_function():
    return MyModel()

def GetInput():
    # Generate random token indices matching the model's expected input
    return torch.randint(0, 1000, (1, 128), dtype=torch.long)

