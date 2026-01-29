# torch.rand(B, S, dtype=torch.long)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, vocab_size=10, rnn_dims=512):
        super().__init__()
        self.word_embeds = nn.Embedding(vocab_size, rnn_dims)
        self.emb_drop = nn.Dropout(0.1)
        self.rnn = nn.LSTM(
            input_size=rnn_dims,
            hidden_size=rnn_dims,
            batch_first=True,
            num_layers=2,
            dropout=0.1
        )

    def forward(self, x):
        # Initialize hidden and cell states with batch/device matching input
        h0 = torch.zeros(2, x.size(0), self.rnn.hidden_size, device=x.device)
        c0 = torch.zeros(2, x.size(0), self.rnn.hidden_size, device=x.device)
        embeds = self.emb_drop(self.word_embeds(x))
        out, (hn, cn) = self.rnn(embeds, (h0, c0))
        return (hn, cn)  # Return LSTM hidden states as per original code

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randint(0, 10, (1, 3), dtype=torch.long)

