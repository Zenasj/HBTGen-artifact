# torch.rand(B, 16000, dtype=torch.float32)  # Inferred input shape from librosa.load(sr=16000)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16000, 10)  # Example layer matching audio sample length

    def forward(self, x):
        return self.linear(x)

def my_model_function():
    return MyModel()

def GetInput():
    B = 4  # Example batch size
    return torch.rand(B, 16000, dtype=torch.float32)

