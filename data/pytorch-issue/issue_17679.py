# torch.randint(0, 10, (B,), dtype=torch.long)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.emb = nn.Embedding(10, 10)  # Matches the embedding size in the reproduce example
        self.emb2 = nn.Embedding(10, 10)  # Second embedding for parameter group addition

    def forward(self, x):
        # Sum of embeddings to mimic the loss computation in the reproduce code
        return self.emb(x).sum() + self.emb2(x).sum()

def my_model_function():
    return MyModel()

def GetInput():
    # Generate random indices for embeddings (shape B,)
    return torch.randint(0, 10, (5,), dtype=torch.long)  # B=5 as minimal test case

