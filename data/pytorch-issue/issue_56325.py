# torch.rand(1, 1, 30522, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(MyModel, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(output_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        # Initialize hidden state as a buffer with requires_grad=False
        self.register_buffer('hidden', torch.rand(1, 1, hidden_size))

    def forward(self, input):
        # input shape: (batch, seq_len, vocab_size)
        output, self.hidden = self.gru(input, self.hidden)
        # Extract last output step (seq_len=1)
        output = self.softmax(self.out(output[:, -1]))
        return output

def my_model_function():
    return MyModel(1536, 30522)

def GetInput():
    return torch.rand((1, 1, 30522), dtype=torch.float32)

