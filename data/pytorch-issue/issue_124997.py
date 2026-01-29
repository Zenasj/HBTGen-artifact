# torch.randint(30522, (128, 512), dtype=torch.long)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Mimic BERT's embedding layer (vocab_size=30522, hidden_size=768)
        self.embedding = nn.Embedding(30522, 768)
        # Add a simple layer to make the model executable
        self.linear = nn.Linear(768, 768)

    def forward(self, input_ids):
        # Simple forward pass for demonstration
        x = self.embedding(input_ids)
        return self.linear(x)

def my_model_function():
    # Initialize a dummy BERT-like model
    return MyModel()

def GetInput():
    # Generate input tensor matching BERT's expected input shape and dtype
    return torch.randint(30522, (128, 512), dtype=torch.long)

