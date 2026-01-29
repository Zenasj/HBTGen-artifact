# torch.rand(B, L, dtype=torch.double)  # e.g., (3, 999) for test input
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 51, dtype=torch.double)
        self.lstm2 = nn.LSTMCell(51, 51, dtype=torch.double)
        self.linear = nn.Linear(51, 1, dtype=torch.double)

    def forward(self, input, future=0):
        outputs = []
        h_t = torch.zeros(input.size(0), 51, dtype=torch.double, device=input.device)
        c_t = torch.zeros(input.size(0), 51, dtype=torch.double, device=input.device)
        h_t2 = torch.zeros(input.size(0), 51, dtype=torch.double, device=input.device)
        c_t2 = torch.zeros(input.size(0), 51, dtype=torch.double, device=input.device)

        for input_t in input.chunk(input.size(1), dim=1):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs.append(output)
        for _ in range(future):
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs.append(output)
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(3, 999, dtype=torch.double)  # Matches test input shape (3 samples, 999 time steps)

