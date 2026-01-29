# torch.randint(0, 30522, (64, 7), dtype=torch.long)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(30522, 768)  # BERT vocab size and hidden size
        self.classifier = nn.Linear(768, 2)  # Sequence classification head

    def forward(self, input_ids):
        # Simplified BERT forward pass (embedding + mean pooling + classifier)
        embeddings = self.embedding(input_ids)
        pooled = embeddings.mean(dim=1)  # Pool across sequence length
        return self.classifier(pooled)

def my_model_function():
    return MyModel()

def GetInput():
    # Generate random input matching BERT input requirements
    return torch.randint(0, 30522, (64, 7), dtype=torch.long)

