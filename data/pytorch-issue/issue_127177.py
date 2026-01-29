# torch.rand(B, 1, 100, dtype=torch.float32)  # Inferred input shape (batch, seq_len, features)
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.lstm = nn.LSTM(input_size=100, hidden_size=64, batch_first=True)
        encoder_layer = TransformerEncoderLayer(d_model=64, nhead=2)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=1)
        self.fc = nn.Linear(64, 4)  # Output 4 classes (0-3)
        self.hidden = None
        self.cell = None

    def reset_states(self):
        self.hidden = None
        self.cell = None

    def forward(self, x):
        batch_size = x.size(0)
        # Initialize hidden/cell states if not provided
        h0 = torch.zeros(1, batch_size, 64, device=x.device)
        c0 = torch.zeros(1, batch_size, 64, device=x.device)
        if self.hidden is not None and self.cell is not None:
            h0, c0 = self.hidden, self.cell
        lstm_out, (self.hidden, self.cell) = self.lstm(x, (h0, c0))
        
        # Transformer expects (seq_len, batch, features)
        transformer_in = lstm_out.permute(1, 0, 2)
        transformer_out = self.transformer(transformer_in)
        
        # Take last LSTM output for classification
        # (could also average/concatenate Transformer outputs)
        final_out = transformer_out[-1]  # Take last sequence element
        return self.fc(final_out)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1, 100, dtype=torch.float32)  # (batch=1, seq_len=1, features=100)

