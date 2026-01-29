# torch.randint(5, (B, 10), dtype=torch.long).cuda()  # Inferred input shape (batch_size, sequence_length)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.EmbeddingBag(10, 10, sparse=True)  # Matches the reported model structure

    def forward(self, x):
        return self.embedding(x)

def my_model_function():
    # Initialize model on CUDA as in the original setup
    return MyModel().cuda()

def GetInput():
    # Generate input matching EmbeddingBag requirements (long dtype for indices)
    return torch.randint(5, (10, 10), dtype=torch.long).cuda()  # Matches original input shape (10x10)

