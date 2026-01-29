# torch.rand(16, 5, 1, dtype=torch.float)  # batch_size=16, sequence_length=5, input_dim=1
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, hidden_size, num_layers, output_size, input_dim):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_size, 
                           num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

def my_model_function():
    return MyModel(hidden_size=1, num_layers=1, output_size=1, input_dim=1)

def GetInput():
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    return torch.rand(16, 5, 1, dtype=torch.float, device=device)

