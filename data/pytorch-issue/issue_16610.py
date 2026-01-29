# torch.rand(B, 32, dtype=torch.float)  # Latent vector input (batch_size, latent_size=32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, img_channels=3, latent_size=32):
        super(MyModel, self).__init__()
        self.latent_size = latent_size
        self.img_channels = img_channels
        self.fc1 = nn.Linear(latent_size, 1024)
        self.deconv1 = nn.ConvTranspose2d(1024, 128, 5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, img_channels, 6, stride=2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = x.unsqueeze(-1).unsqueeze(-1)  # Reshape to (B, 1024, 1, 1)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        reconstruction = torch.sigmoid(self.deconv4(x))
        return reconstruction

def my_model_function():
    # Initialize with img_channels=3 and latent_size=32 as in original code
    return MyModel(3, 32)

def GetInput():
    # Generate random latent vectors with batch size 32 (matches original example's mu shape)
    B = 32
    return torch.rand(B, 32, dtype=torch.float)

