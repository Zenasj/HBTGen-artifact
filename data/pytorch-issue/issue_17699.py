# torch.rand(5, 10, dtype=torch.int64) ← Add a comment line at the top with the inferred input shape
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.embedding = nn.Embedding(10, 20, padding_idx=0)
        self.embedding.half()  # Convert to half precision

    def forward(self, x):
        return self.embedding(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randint(0, 10, size=(5, 10), dtype=torch.int64)

