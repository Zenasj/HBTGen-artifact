# torch.rand(B, 3, dtype=torch.long)  # Input shape is (batch_size, 3) with Long type
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding1 = nn.Embedding(num_embeddings=4, embedding_dim=8)
        self.embedding2 = nn.Embedding(num_embeddings=4, embedding_dim=8)
        self.embedding3 = nn.Embedding(num_embeddings=4, embedding_dim=8)
        self.stacked_layer = nn.Sequential(
            nn.Linear(in_features=24, out_features=1052),
            nn.ReLU(),
            nn.BatchNorm1d(1052),
            nn.Linear(in_features=1052, out_features=526),
            nn.ReLU(),
            nn.BatchNorm1d(526),
            nn.Linear(in_features=526, out_features=256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(in_features=64, out_features=4)
        )

    def forward(self, x):
        # Split input into 3 separate features for embeddings
        emb1 = self.embedding1(x[:, 0].long())
        emb2 = self.embedding2(x[:, 1].long())
        emb3 = self.embedding3(x[:, 2].long())
        embedded_features = torch.cat([emb1, emb2, emb3], dim=1)
        return self.stacked_layer(embedded_features)

def my_model_function():
    return MyModel()

def GetInput():
    # Generate random input with correct shape and dtype
    B = 32  # Example batch size
    return torch.randint(0, 4, (B, 3), dtype=torch.long)

