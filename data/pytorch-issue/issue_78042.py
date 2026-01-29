import torch
import torch.nn as nn

# torch.randint(0, 30522, (B, 512), dtype=torch.long)  # BERT input shape: (batch, sequence_length)
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(30522, 768)  # BERT's vocabulary size and hidden size
        self.pooler = nn.Linear(768, 768)  # Mimics BERT's pooler layer

    def forward(self, x):
        embedded = self.embedding(x)
        first_token = embedded[:, 0]  # Extract [CLS] token embedding
        return self.pooler(first_token)

def my_model_function():
    return MyModel()

def GetInput():
    # Generate random token indices matching BERT's expected input
    batch_size = 32  # Original issue's batch size
    seq_length = 512  # Typical BERT sequence length
    return torch.randint(0, 30522, (batch_size, seq_length), dtype=torch.long)

