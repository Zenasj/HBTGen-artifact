# torch.rand(B, C, H, W, dtype=torch.float32)  # Inferred input shape: Batch, Channels, Height, Width
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_a = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.model_b = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.MaxPool2d(2)
        )
        # Encapsulate models to compare outputs using torch.allclose (as per testing discussion)

    def forward(self, x):
        out_a = self.model_a(x)
        out_b = self.model_b(x)
        # Return comparison result as a tensor (1.0 if close, 0.0 otherwise)
        return torch.tensor(
            torch.allclose(out_a, out_b, atol=1e-5, rtol=1e-5),
            dtype=torch.float32
        ).unsqueeze(0)  # Ensure tensor shape compatibility

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 32, 32, dtype=torch.float32)  # Random input matching assumed shape

