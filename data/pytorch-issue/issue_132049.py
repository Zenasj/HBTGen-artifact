# torch.rand(B, 3, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)  # Matches input shape from example (3 features)

    def forward(self, x):
        return self.linear(x)

def my_model_function():
    model = MyModel()
    # Initialize weights for reproducibility (optional)
    with torch.no_grad():
        model.linear.weight.fill_(1.0)
        model.linear.bias.fill_(0.0)
    return model

def GetInput():
    # Returns a batch of 5 samples with 3 features each
    return torch.rand(5, 3, dtype=torch.float32)

