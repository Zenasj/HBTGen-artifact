# torch.rand(B, 32, dtype=torch.float)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = GeneratorNetwork()
        self.discriminator = DiscriminatorNetwork()

    def forward(self, x, mode='generator'):
        if mode == 'generator':
            return self.generator(x)
        elif mode == 'discriminator':
            return self.discriminator(x)
        else:
            raise ValueError("Invalid mode. Use 'generator' or 'discriminator'.")

class GeneratorNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(32, 2)

    def forward(self, x):
        return self.layer(x)

    def parameters(self, recurse: bool = True):
        return self.layer.parameters(recurse=recurse)

class DiscriminatorNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(32, 2)

    def forward(self, x):
        return self.layer(x)

    def parameters(self, recurse: bool = True):
        return self.layer.parameters(recurse=recurse)

def my_model_function():
    # Returns an instance of MyModel with default initialization
    return MyModel()

def GetInput():
    # Generates a random input tensor of shape (batch_size, 32)
    B = 2  # Example batch size (matches user's minimal example)
    return torch.rand(B, 32, dtype=torch.float)

