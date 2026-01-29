# torch.rand(B, 101467, dtype=torch.float32) ← Input shape inferred from user's code
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Simulated feature extractor based on typical Hubert architecture
        self.conv1 = nn.Conv1d(1, 512, kernel_size=10, stride=320)  # Downsamples time dimension
        self.conv2 = nn.Conv1d(512, 1024, kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        
        # Placeholders for post-convolution layers (exact layers unknown)
        self.fc = nn.Linear(1024, 1024)  # Mimic transformer input
        
        # The problematic reshape is dynamically computed here
        # Actual reshape logic inferred from error dimensions (1,316,1024 → 1,49,16,64)
        # Using view with dynamic dimensions to avoid static shape dependency
        # Note: 16*64 = 1024 (feature dimension) so reshape is valid if T' is divisible by target factors
        # Using dynamic reshape to avoid hardcoding problematic shape
        
    def forward(self, x):
        # Input shape: (B, T) → (B, 1, T) for Conv1d
        x = x.unsqueeze(1)  # Add channel dimension
        
        # Feature extraction layers
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        
        # Permute to (B, T', C) format
        x = x.permute(0, 2, 1)  # (B, T', 1024)
        
        # Dynamically reshape to (B, T', 16, 64) using existing dimensions
        # 1024 = 16*64 → valid reshape along feature dimension
        x = x.view(x.size(0), x.size(1), 16, 64)
        
        # Add minimal post-reshape processing (placeholder)
        x = self.fc(x.view(x.size(0), x.size(1), -1))  # Restore to (B, T', 1024)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    # Generate input matching (B, T) = (1, 101467) as per user's code
    return torch.rand(1, 101467, dtype=torch.float32)

