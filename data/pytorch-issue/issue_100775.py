# torch.rand(10, 32, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MyModel, self).__init__()
        self.query = nn.Linear(input_size, hidden_size)
        self.key = nn.Linear(input_size, hidden_size)
        self.scale_factor = nn.Parameter(torch.tensor([-1.7197e+14]))

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores_scaled = scores.mul(self.scale_factor)
        t = scores_scaled.softmax(dim=-1)
        return t

def my_model_function():
    # Initialize with input_size=32 and hidden_size=1 as per the original example
    return MyModel(input_size=32, hidden_size=1)

def GetInput():
    # Generate a random tensor matching the input shape (10, 32)
    return torch.randn(10, 32)

