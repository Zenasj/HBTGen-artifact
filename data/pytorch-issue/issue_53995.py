# torch.rand(6, 3, 10, dtype=torch.float32)  # input_sequence shape (seq_len, batch_size, input_size)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm_cell = nn.LSTMCell(10, 20)  # input_size=10, hidden_size=20
    
    def forward(self, inputs):
        input_sequence, initial_h, initial_c = inputs
        outputs = []
        h, c = initial_h, initial_c
        for x in input_sequence:
            h, c = self.lstm_cell(x, (h, c))
            outputs.append(h)
        return torch.stack(outputs), h, c

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a tuple (input_sequence, hx, cx) with correct shapes
    input_sequence = torch.randn(6, 3, 10)  # 6 time steps, batch 3, input size 10
    hx = torch.randn(3, 20)  # batch 3, hidden size 20
    cx = torch.randn(3, 20)
    return (input_sequence, hx, cx)

