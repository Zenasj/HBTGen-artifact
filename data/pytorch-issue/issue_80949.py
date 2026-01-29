# torch.rand(B, 1, 28, 28, dtype=torch.float32)
import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x):
        return self.l1(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

    def forward(self, x):
        return self.l1(x)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x_flat = x.view(x.size(0), -1)  # Flatten input tensor
        z = self.encoder(x_flat)
        x_hat = self.decoder(z)
        return x_hat

def my_model_function():
    return MyModel()

def GetInput():
    batch_size = 4  # Arbitrary batch size, can be adjusted
    return torch.rand(batch_size, 1, 28, 28, dtype=torch.float32)

