# torch.rand(S, B, 5, dtype=torch.float32)  # S: sequence length, B: batch size
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTMCell(input_size=5, hidden_size=10)
    
    def forward(self, x):
        batch_size = x.size(1)
        hx = torch.rand(batch_size, self.rnn.hidden_size, device=x.device)
        cx = torch.rand(batch_size, self.rnn.hidden_size, device=x.device)
        step_output = None
        for i in range(x.size(0)):
            hx, cx = self.rnn(x[i], (hx, cx))
            current_output = hx
            if step_output is None:
                step_output = current_output
            else:
                step_output = torch.cat((step_output, current_output), dim=0)
        return step_output

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(10, 1, 5)

