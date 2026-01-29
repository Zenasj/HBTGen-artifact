# torch.rand(B, 28, 28, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, input_dim=28, hidden_dim=100, layer_dim=1, output_dim=10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=x.device, requires_grad=True)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=x.device, requires_grad=True)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        return self.fc(out[:, -1, :])

def my_model_function():
    return MyModel()

def GetInput():
    B = 4  # Example batch size; adjust as needed
    return torch.rand(B, 28, 28, dtype=torch.float32)

