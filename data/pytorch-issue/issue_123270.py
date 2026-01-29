# torch.rand(B, 1, 86, 140, dtype=torch.float32)

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
        self.audio_pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.fc1 = nn.Linear(92736, 64)
        self.fc2 = nn.Linear(64, 2)
    
    def forward(self, x):
        x = self.audio_conv1(x)
        x = self.audio_pool1(x)
        x = x.view(x.size(0), -1)  # Flatten for FC layers
        x = self.fc1(x)
        x = self.fc2(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1, 86, 140, dtype=torch.float32)

