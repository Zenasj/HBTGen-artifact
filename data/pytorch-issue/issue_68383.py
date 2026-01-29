# torch.randint(0, 5000, (16,), dtype=torch.long)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 10)  # Matches input shape (B,1) after view()

    def forward(self, x):
        # Reshape 1D input to (batch_size, 1) for linear layer
        x = x.view(-1, 1).float()  
        return self.fc(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Matches DataLoader batch_size=16 and dataset range(5000)
    return torch.randint(0, 5000, (16,), dtype=torch.long)

