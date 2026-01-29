# torch.rand(B, C, H, W, dtype=...)  # Inferred input shape: (seq_len, batch_size, input_size)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(MyModel, self).__init__()
        self.lstm = self.lstm_spectral_norm(input_size, hidden_size, n_layers)
    
    def lstm_spectral_norm(self, input_size, hidden_size, n_layers=1):
        lstm = nn.LSTM(input_size, hidden_size, n_layers)
        name_pre = 'weight'
        for i in range(n_layers):
            name = name_pre + '_hh_l' + str(i)
            lstm = torch.nn.utils.spectral_norm(lstm, name)
            name = name_pre + '_ih_l' + str(i)
            lstm = torch.nn.utils.spectral_norm(lstm, name)
        return lstm
    
    def forward(self, x):
        output, _ = self.lstm(x)
        return output

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel(input_size=128, hidden_size=128)

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    seq_len = 3
    batch_size = 32
    input_size = 128
    return torch.randn(seq_len, batch_size, input_size, device="cuda" if torch.cuda.is_available() else "cpu")

