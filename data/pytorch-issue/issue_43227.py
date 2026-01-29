# torch.rand(B, C, H, W, dtype=torch.float)  # Input shape: batch, channels, sequence_length, additional dimension (e.g., 1)
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.rnn = nn.LSTM(input_size=1, hidden_size=5, batch_first=True)  # Input features: 1 (from C*W=1*1)

    def forward(self, x):
        # Reshape input to (batch, sequence_length, features)
        B, C, H, W = x.size()
        features = C * W
        x_reshaped = x.view(B, H, features)  # (B, H, C*W)
        
        # Compute sequence lengths (non-zero elements along the sequence dimension)
        mask = (x_reshaped != 0).any(dim=2)  # (B, H) mask for non-zero positions
        lengths = mask.sum(dim=1).to(torch.int64)  # (B,)
        lengths_cpu = lengths.to('cpu')  # Convert lengths to CPU as required
        
        # Pack the sequence (input remains on GPU, lengths on CPU)
        packed = pack_padded_sequence(x_reshaped, lengths_cpu, batch_first=True, enforce_sorted=False)
        packed_out, (h, c) = self.rnn(packed)  # Process via LSTM
        return h  # Return hidden state for demonstration

def my_model_function():
    return MyModel()

def GetInput():
    # Generate input matching the issue's example (3x3 padded sequences) with arbitrary batch=2
    B = 2  # Example batch size (adjustable)
    C, H, W = 1, 3, 1  # Matches input shape (B, C, H, W) with sequence length H=3
    input_tensor = torch.rand(B, C, H, W, dtype=torch.float, device='cuda')
    return input_tensor

