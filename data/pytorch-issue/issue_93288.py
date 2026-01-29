# torch.rand(8, 1024, dtype=torch.int64, device='cuda')  # Input shape
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Reproduces the embedding layer configuration from the minified repro
        self.embedding = nn.Embedding(
            num_embeddings=32128,  # Vocabulary size from args[1][0][0]
            embedding_dim=512,     # Embedding dimension from args[1][0][1]
            padding_idx=None,
            max_norm=None,
            norm_type=2.0,
            scale_grad_by_freq=False,
            sparse=False
        )

    def forward(self, input_ids):
        # View operation from the original forward method
        view = input_ids.view(-1, 1024)
        # Embedding computation using the model's parameters
        embedding = self.embedding(view)
        # Reproduces the fixed output tensor creation
        ones = torch.ones(8, 1024, device=input_ids.device)
        return ones, embedding

def my_model_function():
    # Initialize model on CUDA and return
    return MyModel().cuda()

def GetInput():
    # Generate random input matching the expected shape and dtype
    return torch.randint(
        low=0,
        high=32128,  # Matches embedding num_embeddings
        size=(8, 1024),
        dtype=torch.int64,
        device='cuda'
    )

