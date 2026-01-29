# torch.randint(0, 50000, (B, 150), dtype=torch.long)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.max_features = 50000  # Vocabulary size
        self.embedding_dim = 128    # Embedding size
        self.hidden_size = 128      # Hidden layer size
        self.embedding = nn.Embedding(
            num_embeddings=self.max_features,
            embedding_dim=self.embedding_dim
        )
        self.gru = nn.GRU(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        self.l_out = nn.Linear(self.hidden_size, 2)  # Output layer for 2 classes

    def forward(self, x):
        embeds = self.embedding(x)
        batch_size = x.size(0)
        h0 = torch.zeros(1, batch_size, self.hidden_size, device=x.device)
        output, _ = self.gru(embeds, h0)
        last_timestep = output[:, -1, :]
        return self.l_out(last_timestep)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randint(
        0, 50000,
        (64, 150),
        dtype=torch.long
    )

