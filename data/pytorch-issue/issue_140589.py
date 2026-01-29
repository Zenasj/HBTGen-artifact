# torch.rand(B, 4096, dtype=torch.int64)  # Input shape: batch_size x sequence_length (assuming token IDs)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # NV-Embed-v1 stub implementation: typical transformer-based encoder
        self.embedding_dim = 512  # Example dimension matching HuggingFace model specs
        self.token_embeddings = nn.Embedding(num_embeddings=30522, embedding_dim=self.embedding_dim)  # BERT-like vocab size
        self.position_embeddings = nn.Embedding(4096, self.embedding_dim)  # Max position embeddings
        # Simplified transformer layers (actual model has 24 layers but we use 1 for stub)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=8)
        self.layernorm = nn.LayerNorm(self.embedding_dim)
    
    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
        embeddings = self.token_embeddings(input_ids) + self.position_embeddings(position_ids)
        # Apply single transformer layer as stub
        transformer_output = self.transformer_layer(embeddings.transpose(0, 1)).transpose(0, 1)
        return self.layernorm(transformer_output).mean(dim=1)  # Mean pooling for embedding

def my_model_function():
    return MyModel()

def GetInput():
    # Generate random token IDs matching NV-Embed-v1 input requirements
    B = 2  # Batch size from original example's two queries
    return torch.randint(0, 30522, (B, 4096), dtype=torch.int64)

