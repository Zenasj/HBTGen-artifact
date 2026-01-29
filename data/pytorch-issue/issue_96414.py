# torch.rand(1, 4, dtype=torch.int64, device="cuda:0")
import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Define TransformerEncoderLayer with d_model=4, nhead=1, dim_feedforward=16
        layer = TransformerEncoderLayer(d_model=4, nhead=1, dim_feedforward=16)
        self.encoder = TransformerEncoder(layer, num_layers=1)
        # Embedding layer with 16 vocab entries and 4-dimensional embeddings
        self.embedding = nn.Embedding(num_embeddings=16, embedding_dim=4, padding_idx=0)
    
    def forward(self, input):
        embedded = self.embedding(input)
        return self.encoder(embedded)

def my_model_function():
    # Initialize model in eval mode and move to CUDA
    model = MyModel()
    model.eval()
    model.cuda()
    return model

def GetInput():
    # Generate random integer tensor matching the embedding's requirements
    return torch.randint(1, 16, (1, 4), dtype=torch.int64, device="cuda:0")

