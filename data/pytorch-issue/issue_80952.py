# torch.rand(B, 10, dtype=torch.float32)  # Add a comment line at the top with the inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
        self.ln = nn.LayerNorm(5, elementwise_affine=True)

    def forward(self, x):
        x = self.fc(x)
        x = self.ln(x)
        return x

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    model = MyModel()
    model.fc.weight.requires_grad = False
    model.fc.bias.requires_grad = False
    return model

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    B = 2  # Batch size
    return torch.randn(B, 10, dtype=torch.float32)

