# torch.randint(0, 100, (1, 128), dtype=torch.long)  # Input shape for BERT (batch=1, seq_len=128)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Mimic BERT's embedding layer (vocab size 30522, hidden size 768)
        self.embedding = nn.Embedding(30522, 768)
        # Dummy layer to trigger computation (matches error scenario)
        self.linear = nn.Linear(768, 768)
        
    def forward(self, input_ids):
        # Simulate BERT's forward path (input_ids -> embeddings -> linear layer)
        x = self.embedding(input_ids)
        return self.linear(x)

def my_model_function():
    # Returns a BERT-like model instance
    return MyModel()

def GetInput():
    # Generates input tensor matching BERT's expected input shape and dtype
    return torch.randint(0, 100, (1, 128), dtype=torch.long)

