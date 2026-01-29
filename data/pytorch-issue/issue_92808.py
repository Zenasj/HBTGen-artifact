# torch.randint(0, 30522, (1, 128), dtype=torch.long)  # BERT input_ids shape and dtype
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Mimicking BERT's embedding layer (vocab size 30522, hidden size 768)
        self.embedding = nn.Embedding(30522, 768)
        # Simple linear layer as placeholder for BERT's encoder
        self.linear = nn.Linear(768, 768)
    
    def forward(self, input_ids):
        # Dummy forward pass to replicate model structure
        embedded = self.embedding(input_ids)
        return self.linear(embedded)

def my_model_function():
    # Initialize a minimal BERT-like model
    return MyModel()

def GetInput():
    # Generate random input_ids matching BERT's expected input shape
    return torch.randint(0, 30522, (1, 128), dtype=torch.long)

