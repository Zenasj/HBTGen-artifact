# torch.rand(1, 20, 1, dtype=torch.float32)  # Add a comment line at the top with the inferred input shape

import torch
import torch.nn as nn
import numpy as np

class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hs=None):
        # Forward pass through RNN layer
        rnn_out, hidden = self.rnn(x, hs)
        
        # Reshape RNN output for input to FC layer
        rnn_out = rnn_out.view(-1, self.hidden_size)
        
        # Forward pass through FC layer
        output = self.fc(rnn_out)
        
        # Return prediction and updated hidden state
        return output, hidden

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    input_size = 1
    hidden_size = 10
    num_layers = 2
    output_size = 1
    model = MyModel(input_size, hidden_size, num_layers, output_size)
    model = model.to(dtype=torch.float32)  # Ensure the model uses float32
    return model

def GetInput():
    # Generate sequence `x` of 20 data points
    seq_len = 20
    time = np.linspace(0, np.pi, seq_len + 1)
    data = np.sin(time)
    x = data[:-1].reshape((seq_len, 1))
    
    # Convert to tensor and ensure the dtype is float32
    test_input = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    return test_input

