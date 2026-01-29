# torch.rand(16, 6000, 256, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.h_size = 600  # Hidden layer size from original code
        self.e_size = 900  # Embedding size from original code
        self.l1 = nn.Linear(256, self.e_size)  # Input layer
        # Bidirectional LSTM with 2 layers, matches original configuration
        self.lstm = nn.LSTM(
            self.e_size, self.h_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        # Output layer takes concatenated hidden states from bidirectional LSTM
        self.l2 = nn.Linear(self.h_size * 2, 300)

    def forward(self, input):
        batch_size = input.size(0)
        # Initialize hidden states on same device as input
        hidden = (
            torch.zeros(2*2, batch_size, self.h_size, device=input.device),
            torch.zeros(2*2, batch_size, self.h_size, device=input.device)
        )
        # Flatten batch and sequence dimensions for linear layer
        l1_out = F.relu(self.l1(input.view(-1, 256)))
        # Restore sequence dimension for LSTM
        l1_out = l1_out.view(batch_size, -1, self.e_size)
        lstm_out, _ = self.lstm(l1_out, hidden)
        # Flatten again for final linear layer
        l2_out = F.relu(self.l2(lstm_out.contiguous().view(-1, self.h_size*2)))
        return l2_out

def my_model_function():
    # Create model instance with default initialization
    return MyModel()

def GetInput():
    # Create input tensor matching expected shape and dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.rand(16, 6000, 256, dtype=torch.float32, device=device)

