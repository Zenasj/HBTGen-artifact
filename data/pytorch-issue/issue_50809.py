# torch.rand(B, S, C, dtype=torch.float32)  # B=batch, S=sequence length, C=feature size
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, input_size=100, hidden_size=50):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(input_size, 2 * hidden_size)  # Produces tensor for chunking
        
    def forward(self, x):
        y = self.linear(x)
        z_gate, f_gate = y.chunk(2, dim=2)  # Split into views
        z_gate.tanh_()  # In-place operation on view (problematic)
        f_gate.sigmoid_()  # Another in-place operation on view
        return z_gate, f_gate  # Return tuple to trigger autograd issues

def my_model_function():
    # Example initialization with inferred parameters
    return MyModel()

def GetInput():
    # Generate input matching (batch, sequence_length, features)
    return torch.rand(2, 32, 100, dtype=torch.float32)  # B=2, S=32, C=100

