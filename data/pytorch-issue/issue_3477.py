# torch.rand(10, 32, 50, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.rnn = nn.RNN(input_size=50, hidden_size=50, num_layers=2)
    
    def forward(self, x):
        # Forward pass through the RNN layer
        output, _ = self.rnn(x)
        return output

def my_model_function():
    # Returns an instance of MyModel with default initialization
    return MyModel()

def GetInput():
    # Generates a random input tensor matching the model's expected shape
    return torch.randn(10, 32, 50, dtype=torch.float32)

