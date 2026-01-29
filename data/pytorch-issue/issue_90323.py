# torch.rand(1, 3, 224, 224, dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu")
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        
    def forward(self, x):
        # Dynamo error occurs when evaluating this condition
        if torch.backends.cudnn.is_acceptable(x):
            return self.conv(x)
        else:
            # Fallback path when CUDNN is not acceptable
            return nn.functional.conv2d(x, self.conv.weight, self.conv.bias, padding=1)

def my_model_function():
    # Initialize with default parameters
    return MyModel()

def GetInput():
    # Generate input matching the model's expected dimensions
    batch_size = 1
    channels = 3
    spatial_dim = 224
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.rand(batch_size, channels, spatial_dim, spatial_dim, 
                     dtype=torch.float32, device=device)

