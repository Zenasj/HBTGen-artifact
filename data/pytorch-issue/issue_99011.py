# torch.randint(0, 10000, (B, S), dtype=torch.long)  # Input shape: Batch x Sequence Length
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=10000, embedding_dim=512)
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.fc = nn.Linear(512, 10000)  # Output layer with shared weights
        self.fc.weight = self.embedding.weight  # Share the embedding weights

    def forward(self, x):
        x = self.embedding(x)  # Shape: (batch, seq_len, 512)
        x = self.transformer(x)  # Shape remains (batch, seq_len, 512)
        x = x.mean(dim=1)  # Global average pooling over sequence length
        return self.fc(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Default input: batch size 2, sequence length 16 (typical for distributed tests)
    return torch.randint(0, 10000, (2, 16), dtype=torch.long)

