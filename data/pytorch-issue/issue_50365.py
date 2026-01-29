# torch.rand(3, 1, 8, dtype=torch.float)
import torch
import torch.nn as nn

class MyLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MyLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Linear(input_size, 4 * hidden_size)
        self.weight_hh = nn.Linear(hidden_size, 4 * hidden_size)

    def forward(self, input, state):
        hx, cx = state
        gates = self.weight_ih(input) + self.weight_hh(hx)
        ingate, forgetgate, cellgate, outgate = torch.split(gates, self.hidden_size, dim=1)
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)
        cy = forgetgate * cx + ingate * cellgate
        hy = outgate * torch.tanh(cy)
        return hy, (hy, cy)

class MyModel(nn.Module):
    def __init__(self, input_size=8, hidden_size=20):
        super(MyModel, self).__init__()
        self.lstm = MyLSTMCell(input_size, hidden_size)
        self.hidden_size = hidden_size
        # Initialize initial states as buffers (fixed)
        self.register_buffer('hx_initial', torch.randn(1, hidden_size))
        self.register_buffer('cx_initial', torch.randn(1, hidden_size))

    def forward(self, x):
        batch_size = x.size(1)
        hx = self.hx_initial.expand(batch_size, -1).clone()
        cx = self.cx_initial.expand(batch_size, -1).clone()
        output = []
        for i in range(x.size(0)):
            hy, (hx, cx) = self.lstm(x[i], (hx, cx))
            output.append(hy)
        return hx  # Returns the last hidden state

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(3, 1, 8, dtype=torch.float)

