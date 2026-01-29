# torch.rand(B, 28, 28, dtype=torch.float)  # MNIST input: batch x sequence_length x input_size
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, input_size=28, hidden_size=128, num_layers=2, num_classes=10):
        super(MyModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        self.gru.flatten_parameters()
        # GRU does not require cell state (c0), so only h0 is initialized
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x)  # out: (batch_size, seq_length, hidden_size)
        out = self.fc(out[:, -1, :])  # Last time step's output
        return out

def my_model_function():
    # Initialize with default MNIST parameters
    return MyModel()

def GetInput():
    # Generate random input matching GRU's expected shape (batch, sequence, features)
    batch_size = 100  # Matches issue's batch_size hyperparameter
    return torch.rand(batch_size, 28, 28, dtype=torch.float)

