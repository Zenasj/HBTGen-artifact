# torch.randint(0, 30000, (B, S), dtype=torch.long)  # Inferred input shape (B=batch, S=sequence length)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(30000, 768)  # Mimic Llama's embedding layer
        self.fc = nn.Linear(768, 768)          # Example layer with autocast usage
        
    def forward(self, input_ids):
        # Replicate autocast context causing export issues
        with torch.autocast(device_type='cuda', enabled=True):
            x = self.embed(input_ids)
            x = self.fc(x)
            return x

def my_model_function():
    return MyModel()

def GetInput():
    B = 2  # Batch size
    S = 5  # Sequence length
    return torch.randint(0, 30000, (B, S), dtype=torch.long)

