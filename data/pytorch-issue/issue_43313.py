# torch.rand(B, 3, 64, 64, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Example encoder (simplified from user's autoencoder setup)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2),  # Input: (3,64,64) → (16,31,31)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),  # → (32,14,14)
            nn.ReLU(),
        )
        # Example decoder (to reconstruct original image dimensions)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2),  # → (16,28,28)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2),  # → (3,60,60) → padding may be needed
            nn.Sigmoid(),  # Ensure output is 0-1 like input images
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def my_model_function():
    # Returns initialized model with default parameters
    return MyModel()

def GetInput():
    # Returns random input tensor matching model's expected dimensions
    return torch.rand(4, 3, 64, 64, dtype=torch.float32)  # Batch=4, RGB, 64x64 images

