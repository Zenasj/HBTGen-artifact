# torch.randint(0, 30522, (B, 128), dtype=torch.long)  # B: batch size, 128 sequence length
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = nn.Sequential(
            nn.Embedding(30522, 768),  # BERT-like embedding layer
            nn.Linear(768, 768),       # Dummy BERT layer (replace with actual BERT structure if known)
            nn.ReLU()
        )
        self.classifier = nn.Linear(768, 2)  # Example output layer

    def forward(self, x):
        x = self.bert(x)
        return self.classifier(x.mean(dim=1))  # Global average pooling for sequence classification

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randint(0, 30522, (2, 128), dtype=torch.long)

