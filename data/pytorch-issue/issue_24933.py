# torch.rand(B, F, dtype=torch.float32)  # Example input shape (B=32, F=1024)

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_with_logits = nn.BCEWithLogitsLoss()
        self.bce = nn.BCELoss()

    def forward(self, x):
        inputs, targets = x
        loss1 = self.bce_with_logits(inputs, targets)
        loss2 = self.bce(torch.sigmoid(inputs), targets)
        return torch.abs(loss1 - loss2)

def my_model_function():
    return MyModel()

def GetInput():
    B, F = 32, 1024  # Example shape from test script
    inputs = torch.randn(B, F, dtype=torch.float32) * 1000  # Large values to test overflow
    targets = torch.rand(B, F, dtype=torch.float32)
    return (inputs, targets)

