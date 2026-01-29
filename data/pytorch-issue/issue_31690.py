# torch.rand(B, C, H, W, dtype=...)  # (1028, 128, 6, 6)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, output_padding):
        super().__init__()
        
        self.c1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=output_padding)
        self.c2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=output_padding)
        self.c3 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=output_padding)
    
    def forward(self, x):
        x = self.c1(x)
        x = nn.functional.relu(x)
        x = self.c2(x)
        x = nn.functional.relu(x)
        x = self.c3(x)
        x = nn.functional.relu(x)
        return x

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel(output_padding=(0, 0))

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand((1028, 128, 6, 6), device='cuda')

