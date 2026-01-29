# torch.rand(B, 4, 10, dtype=torch.long)  # B=batch, 4=number of attributes, 10=sequence length
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(10000, 128)  # Vocabulary size=10k, embedding_dim=128
        self.fc = nn.Linear(128 * 4, 1)  # 4 attributes, aggregated into a single output
        
    def forward(self, x):
        # x: (B, 4, 10) → word indices for 4 attributes with max length 10
        embedded = self.embedding(x)  # (B,4,10,128)
        pooled = embedded.mean(dim=2)  # Average across sequence length → (B,4,128)
        flattened = pooled.view(pooled.size(0), -1)  # (B, 4*128)
        return self.fc(flattened)

def my_model_function():
    return MyModel()

def GetInput():
    batch_size = 2
    return torch.randint(0, 10000, (batch_size, 4, 10), dtype=torch.long)

