import torch
import torch.nn as nn

# torch.rand(B, 13), torch.ones(B, dtype=torch.int64)*26, torch.randint(0, 100000, (B*26,), dtype=torch.int64)
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Dummy embedding module to simulate 26 features each with 64 dims (total 1664)
        self.embedding = nn.Sequential(
            nn.Linear(1, 26 * 64),  # Dummy layer to generate the correct output shape
            nn.Identity()  # Pass through, actual logic is mocked
        )
        # Dense layers: 13 inputs -> [512, 256, 64] -> 64 output
        self.dense_layers = nn.Sequential(
            nn.Linear(13, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU()
        )
        # Over-arch: (64 + 1664) -> [512, 512, 256, 1] with final sigmoid
        self.over_layers = nn.Sequential(
            nn.Linear(64 + 26*64, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, inputs):
        dense_features, _, _ = inputs  # lengths and values are mocked
        # Mock embedding output (shape B x 1664)
        embeddings = self.embedding(dense_features.new_ones(dense_features.shape[0], 1)).view(dense_features.shape[0], -1)
        dense_out = self.dense_layers(dense_features)
        combined = torch.cat([dense_out, embeddings], dim=1)
        return self.over_layers(combined)

def my_model_function():
    return MyModel()

def GetInput():
    B = 10
    dense = torch.rand(B, 13)
    lengths = torch.full((B,), 26, dtype=torch.int64)
    values = torch.randint(0, 100000, (B*26,), dtype=torch.int64)
    return (dense, lengths, values)

