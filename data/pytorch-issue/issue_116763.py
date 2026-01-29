# torch.randint(0, 1000, (2, 5), dtype=torch.long)  # Example input shape: batch=2, sequence_length=5
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=1000, embedding_dim=64)  # Example embedding dimensions
        self.gru = nn.GRU(input_size=64, hidden_size=128, batch_first=True)   # GRU layer
        
    def forward(self, x):
        emb = self.embedding(x)
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            out, hidden = self.gru(emb)
        return out, hidden

def my_model_function():
    model = MyModel()
    model = model.to(torch.bfloat16)  # Ensure model uses bfloat16 parameters
    return model

def GetInput():
    # Generate random input indices for the embedding layer (batch_size=2, seq_len=5)
    return torch.randint(0, 1000, (2, 5), dtype=torch.long)

