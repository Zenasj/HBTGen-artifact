# torch.rand(20, 1, 100, dtype=torch.float32)  # Example input shape: (seq_len, batch_size, input_size)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, input_size=100, hidden_size=128, output_size=2):
        super(MyModel, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)  # Replaced Softmax + log with LogSoftmax for numerical stability
        
    def forward(self, input_sequence):
        batch_size = input_sequence.size(1)
        hidden = self.initHidden(batch_size)
        for i in range(input_sequence.size(0)):
            combined = torch.cat((input_sequence[i], hidden), dim=1)
            hidden = torch.tanh(self.i2h(combined))  # Added tanh activation for standard RNN behavior
            output = self.i2o(combined)
            output = self.softmax(output)
        return output  # Returns final output (used in loss calculation)
    
    def initHidden(self, batch_size=1):
        # Initialize hidden state on the same device as model parameters
        return torch.zeros(batch_size, self.hidden_size, device=next(self.parameters()).device)

def my_model_function():
    return MyModel()  # Uses default parameters inferred from context

def GetInput():
    # Generates a random input tensor matching the expected dimensions
    seq_len = 20  # Example sequence length (arbitrary choice based on context)
    input_size = 100  # Inferred from character-level input (e.g., one-hot vectors)
    return torch.rand(seq_len, 1, input_size, dtype=torch.float32)

