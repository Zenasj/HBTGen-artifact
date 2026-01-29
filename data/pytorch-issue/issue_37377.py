# torch.rand(B, T, D, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.embedding = nn.Linear(512, 256)  # Input feature dimension from fairseq examples
        self.lstm = nn.LSTM(256, 128, batch_first=True)
        self.fc = nn.Linear(128, 10)  # Example output layer
        
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        return self.fc(x[:, -1, :])  # Output last timestep for classification

def my_model_function():
    # Initialize model with default PyTorch weights
    return MyModel()

def GetInput():
    # Generate random input tensor matching (Batch, Time, Features)
    return torch.rand(8, 128, 512, dtype=torch.float32)  # Batch=8, 128 tokens, 512 features

