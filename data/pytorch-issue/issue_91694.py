# torch.rand(B, C, H, W, dtype=...)  # (batch_size, seq_len, emb_len, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, d_input, n_hidden, n_layers, n_output, dropout, bidirectional=False):
        super(MyModel, self).__init__()
        self.d_input = d_input
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.n_output = n_output
        self.drop = dropout
        self.bidirectional = 1 if not bidirectional else 2
        
        self.rnn = nn.LSTM(input_size=self.d_input,
                           hidden_size=self.n_hidden,
                           num_layers=self.n_layers,
                           batch_first=True,
                           dropout=self.drop,
                           bidirectional=bidirectional)
        
        self.fc = nn.Linear(self.n_hidden * self.bidirectional, self.n_output)
    
    def forward(self, x):
        h0 = torch.zeros(self.n_layers * self.bidirectional, x.size(0), self.n_hidden).to(x.device)
        c0 = torch.zeros(self.n_layers * self.bidirectional, x.size(0), self.n_hidden).to(x.device)
        
        out, _ = self.rnn(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        
        return out

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel(d_input=16, n_hidden=12, n_layers=8, n_output=1, dropout=0.2)

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    batch_size = 64
    seq_len = 32
    emb_len = 16
    return torch.randn((batch_size, seq_len, emb_len)).to(torch.float32)

