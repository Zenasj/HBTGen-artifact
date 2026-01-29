# torch.rand(B, 2, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        inputs, targets = x  # Unpack the input tuple
        return self.loss(inputs, targets)

def my_model_function():
    return MyModel()

def GetInput():
    B, C = 3, 2  # Batch size and number of classes (from issue example)
    inputs = torch.rand(B, C, dtype=torch.float32)
    targets = torch.randint(0, C, (B,), dtype=torch.long)
    return (inputs, targets)  # Returns tuple of inputs and targets

