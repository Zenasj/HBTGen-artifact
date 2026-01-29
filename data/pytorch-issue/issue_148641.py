# torch.rand(B, C, dtype=torch.float32)
import torch
import torch.nn as nn
from torch.utils import checkpoint

class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        def custom_forward(*inputs):
            layer, input_tensor = inputs
            return layer(input_tensor)
        # Use checkpointing on the first layer
        out = checkpoint.checkpoint(custom_forward, self.fc1, x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def my_model_function():
    # Initialize with parameters from test case (input_size=5, hidden_size=10, output_size=5)
    return MyModel(input_size=5, hidden_size=10, output_size=5)

def GetInput():
    # Match input shape (batch_size=4, input_size=5) with requires_grad=True
    return torch.randn(4, 5, requires_grad=True)

