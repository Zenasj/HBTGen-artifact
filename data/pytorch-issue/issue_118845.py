# torch.rand(B, S, input_size, dtype=torch.float32)  # Example input shape: (batch_size, sequence_length, input_features)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)  # Maintains original dim=2 as per reported code

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.softmax(x)
        return x

def my_model_function():
    # Uses default parameters matching input_size=10, hidden_size=20, output_size=5 (arbitrary but consistent)
    return MyModel(input_size=10, hidden_size=20, output_size=5)

def GetInput():
    # Returns 3D tensor (batch, sequence_length, features) to match LogSoftmax(dim=2) requirement
    return torch.rand(2, 3, 10)  # batch=2, sequence_length=3, input_size=10

