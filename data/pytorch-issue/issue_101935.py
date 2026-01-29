# torch.rand(B, S, input_size, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=20, num_layers=2, output_size=1):
        super(MyModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def my_model_function():
    # Returns a model initialized with default parameters
    model = MyModel()
    return model

def GetInput():
    # Returns input tensor matching the LSTM's batch_first=True expectation (batch, seq_len, features)
    batch_size = 5
    seq_len = 3
    input_size = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.randn(batch_size, seq_len, input_size, device=device, dtype=torch.float32)

