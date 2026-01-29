# torch.rand(B, 10, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)  # Example model with 10 input features and 2 classes

    def forward(self, x):
        logits = self.fc(x)
        return torch.argmax(logits, dim=1)  # Returns class indices (long tensor)

def my_model_function():
    return MyModel()

def GetInput():
    B = 4  # Example batch size
    return torch.rand(B, 10, dtype=torch.float32)  # Matches input shape comment

