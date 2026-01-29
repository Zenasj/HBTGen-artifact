# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.bn1d = nn.BatchNorm1d(50)  # Matches C from input shape
        self.bn2d = nn.BatchNorm2d(50)  # Matches C from input shape

    def forward(self, x):
        # Process with BatchNorm2d
        out2d = self.bn2d(x)
        
        # Process with BatchNorm1d (requires reshaping)
        B, C, H, W = x.shape
        x_reshaped = x.view(B * H * W, C)  # Flatten spatial dimensions
        out1d = self.bn1d(x_reshaped)
        
        # Reshape back to original dimensions for comparison
        out1d = out1d.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        
        # Compare outputs using torch.allclose with default tolerances
        is_close = torch.allclose(out1d, out2d, atol=1e-5, rtol=1e-5)
        return torch.tensor([is_close], dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(50, 50, 50, 50, dtype=torch.float32)  # Matches original test shape (B=50, C=50, H=50, W=50)

