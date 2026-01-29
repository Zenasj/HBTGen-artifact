# torch.rand(B, 3, 100, 100, dtype=torch.float32, device='cuda')

import torch
import torch.nn as nn

class EWDNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 100))
        self.fc = nn.Linear(32, 32)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), x.size(1), -1)  # Reshape to (B, 32, 100)
        x = self.fc(x.permute(0, 2, 1)).permute(0, 2, 1)  # Apply linear layer to each position
        return x

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.ewd_net = EWDNet()  # Submodule for network processing

    def forward(self, x):
        # Process through network and compute Cholesky
        z_model = self.ewd_net(x).squeeze(0)
        conv_model = z_model @ z_model.transpose(-1, -2)
        L_model = torch.linalg.cholesky(conv_model)

        # Generate random tensor and compute Cholesky for comparison
        z_random = torch.randn_like(z_model)
        conv_random = z_random @ z_random.transpose(-1, -2)
        L_random = torch.linalg.cholesky(conv_random)

        return L_model, L_random  # Return both outputs for timing comparison

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1, 3, 100, 100, dtype=torch.float32, device='cuda')

