# torch.randint(128, (32,), device=device) ← Add a comment line at the top with the inferred input shape
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.embedding = nn.Embedding(128, 16, max_norm=1)

    def forward(self, x):
        return self.embedding(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch = torch.randint(128, (32,), device=device)
    return batch

