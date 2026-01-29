# torch.randint(0, 150, (B, S), dtype=torch.long)  # B=batch, S=sequence length
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_dimensions, num_layers, cell_type, embedding_dropout, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_size, embedding_dimensions)
        self.embedding_dropout_layer = nn.Dropout(p=embedding_dropout)  # renamed to avoid parameter conflict
        if cell_type == 'LSTM':
            self.rnn = nn.LSTM(input_size=embedding_dimensions, hidden_size=hidden_size, 
                              num_layers=num_layers, dropout=dropout)
        else:
            self.rnn = nn.GRU(input_size=embedding_dimensions, hidden_size=hidden_size, 
                             num_layers=num_layers, dropout=dropout)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.embedding_dropout_layer(x)
        outputs, _ = self.rnn(x)
        return outputs

def my_model_function():
    # Using hyperparameters from the original issue's example
    return MyModel(
        input_size=150,
        hidden_size=256,
        embedding_dimensions=512,
        num_layers=2,
        cell_type='LSTM',
        embedding_dropout=0.0,
        dropout=0.0
    )

def GetInput():
    # Returns a random integer tensor of shape (batch, sequence_length)
    return torch.randint(0, 150, (32, 10), dtype=torch.long)  # B=32, S=10 (example values)

