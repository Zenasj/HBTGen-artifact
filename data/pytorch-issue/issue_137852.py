# torch.rand(B, 1, 28, 28, dtype=torch.float32)  # FashionMNIST input shape (grayscale images)
import torch.nn as nn
import torch

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def my_model_function():
    return MyModel()

def GetInput():
    # Batch size 4 as an example, can be adjusted
    return torch.rand(4, 1, 28, 28, dtype=torch.float32)

