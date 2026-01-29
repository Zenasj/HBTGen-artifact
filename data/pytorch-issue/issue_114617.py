# torch.rand(seq_len, batch_size, input_size, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Initialize RNN with corrected parameter passing (nonlinearity as keyword)
        self.rnn = nn.RNN(input_size=2, hidden_size=3, num_layers=1, nonlinearity='relu')
    
    def forward(self, x):
        # Forward pass returns outputs (ignoring hidden state for simplicity)
        return self.rnn(x)[0]

def my_model_function():
    # Returns properly initialized RNN model
    return MyModel()

def GetInput():
    # Generates input matching RNN's expected dimensions (seq_len, batch, input_size)
    return torch.randn(5, 1, 2, dtype=torch.float32)

