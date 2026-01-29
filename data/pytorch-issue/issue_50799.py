# torch.randint(0, 2, (B,), dtype=torch.long)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Initialize embedding with random weights (as in the original issue)
        self.embedding = nn.Embedding.from_pretrained(torch.rand(2, 3), freeze=True)
    
    def forward(self, x):
        return self.embedding(x)

def my_model_function():
    # Returns the model instance (no additional initialization needed)
    return MyModel()

def GetInput():
    # Generate valid indices (0 or 1) for the embedding layer
    return torch.randint(0, 2, (1,), dtype=torch.long)

