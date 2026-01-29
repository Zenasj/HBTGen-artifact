# torch.rand(B, 100, dtype=torch.long) ← Add a comment line at the top with the inferred input shape
import torch
from torch import nn

class Subsubmodule(nn.Module):
    def __init__(self, input_size, output_size):
        super(Subsubmodule, self).__init__()
        self.emb = nn.Embedding(input_size, output_size)
        
    def forward(self, x):
        return self.emb(x)

class Submodule(nn.Module):
    def __init__(self, input_size, output_size):
        super(Submodule, self).__init__()
        self.emb = Subsubmodule(input_size, output_size)
        
    def forward(self, x):
        return self.emb(x)

class MyModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(MyModel, self).__init__()
        self.encode = Submodule(input_size, output_size)
        self.emb = self.encode.emb.emb  # I want to use a shortcut in some case
        
    def forward(self, x):
        return self.emb(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel(100, 32)

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    B = 4  # Batch size
    return torch.randint(0, 100, (B, 100), dtype=torch.long)

