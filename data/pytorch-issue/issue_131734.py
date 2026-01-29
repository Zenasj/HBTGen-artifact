# torch.randint(0, 30000, (B, 1), dtype=torch.int64)  # Input shape: batch_size × sequence_length=1 (tokenized indices)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(30000, 4096)  # Vocabulary size ~30k, embedding dim 4096 (matches root cause shape)
    
    def forward(self, x):
        x = self.embedding(x)  # Shape (B, 1, 4096) after embedding
        return x.view(x.size(0), -1)  # Triggers stride [0,1] when B=1 (view from (1,1,4096) to (1,4096))

def my_model_function():
    return MyModel()  # Minimal model replicating problematic shape/stride pattern

def GetInput():
    B = 1  # Batch size 1 (critical case), but function works for any B
    return torch.randint(0, 30000, (B, 1), dtype=torch.int64)  # Matches input requirements

