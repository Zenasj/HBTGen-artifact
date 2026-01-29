# torch.rand(B, S, 100, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, input_size=100, hidden_size=50, num_layers=3, num_classes=2):
        super(MyModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Dropout added as per original issue's problematic configuration
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Infer device from input tensor
        device = x.device
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=device)
        c0 = torch.zeros_like(h0)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

def my_model_function():
    # Returns the model instance with parameters matching the issue's example
    return MyModel()

def GetInput():
    # Returns a random input tensor compatible with MyModel's LSTM (B, S, input_size)
    # Uses CUDA to match the original issue's context
    return torch.randn(2, 5, 100, dtype=torch.float32).cuda()

