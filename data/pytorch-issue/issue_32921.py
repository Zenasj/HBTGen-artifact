# torch.randint(0, 10, (B, 10), dtype=torch.long)  # Input shape: batch_size x sequence_length=10, values in [0, 9] (n_vocab=10)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, n_vocab, n_embed, hidden_size, seq_len, num_layers, output_size, drop_prob):
        super().__init__()
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.embedding = nn.Embedding(n_vocab, n_embed)
        self.lstm = nn.LSTM(n_embed, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x, (h0, c0))
        lstm_out = lstm_out.contiguous().view(-1, self.seq_len, 2, self.hidden_size)
        lstm_out_fw = lstm_out[:, -1, 0, :]  # Last step of forward direction
        lstm_out_bw = lstm_out[:, 0, 1, :]   # First step of backward direction (last input)
        lstm_out = torch.cat((lstm_out_fw, lstm_out_bw), dim=-1)
        drop_out = self.dropout(lstm_out)
        logits = self.fc(drop_out)
        return logits

def my_model_function():
    return MyModel(n_vocab=10, n_embed=10, hidden_size=10, seq_len=10, num_layers=10, output_size=10, drop_prob=0.5)

def GetInput():
    return torch.randint(0, 10, (2, 10), dtype=torch.long)

